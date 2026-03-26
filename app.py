from flask import Flask, render_template, request
import pandas as pd, numpy as np, matplotlib, io, base64, warnings
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
warnings.filterwarnings('ignore')

try:
    from prophet import Prophet
    PROPHET_OK = True
except ImportError:
    PROPHET_OK = False

app = Flask(__name__)
COLORES = {'promedio_movil': '#E74C3C', 'ses': '#2980B9', 'prophet': '#27AE60'}

# ── Helpers ───────────────────────────────────────────────────────────────────

def fig_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def errores(real, pred):
    r = pd.Series(real.values)
    p = pd.Series(pred.values[:len(r)] if hasattr(pred, 'values') else list(pred)[:len(r)])
    df = pd.DataFrame({'r': r, 'p': p}).dropna()
    df = df[df.r != 0]
    if df.empty: return {'mape': None, 'mse': None, 'rmse': None}
    e = df.r - df.p
    mse = float((e**2).mean())
    return {k: round(v, 4) for k, v in {
        'mape': float((abs(e / df.r).mean()) * 100),
        'mse': mse, 'rmse': float(np.sqrt(mse))}.items()}

def empaquetar(label, fitted, futuros, err, extra={}):
    return {'label': label, 'fitted': fitted, 'futuros': futuros, 'errores': err,
            'tabla_futuros': [{'periodo': str(i), 'valor': round(v, 2)}
                               for i, v in zip(futuros.index, futuros.values)],
            'extra': extra}

# ── Modelos ───────────────────────────────────────────────────────────────────

def pm(serie, n, fut_idx):
    fitted = serie.shift(1).rolling(n).mean()
    return fitted, pd.Series([round(serie.iloc[-n:].mean(), 4)] * len(fut_idx), index=fut_idx)

def ses(serie, fut_idx):
    res = SimpleExpSmoothing(serie.values, initialization_method='estimated').fit(optimized=True)
    return (pd.Series(res.fittedvalues, index=serie.index),
            pd.Series(res.forecast(len(fut_idx)), index=fut_idx),
            round(res.params['smoothing_level'], 4))

def prophet(serie, fut_idx):
    if not PROPHET_OK: return None, None
    n_hist = len(serie)
    ds = serie.index if isinstance(serie.index, pd.DatetimeIndex) else pd.date_range('2020-01-01', periods=n_hist, freq='MS')
    df_p = pd.DataFrame({'ds': ds, 'y': serie.values})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(df_p)
    h = len(fut_idx)
    fc = m.predict(m.make_future_dataframe(periods=h, freq='MS'))
    # Extraer por posición: primeros n_hist son in-sample, el resto es proyección
    fitted  = pd.Series(fc['yhat'].values[:n_hist], index=serie.index)
    futuros = pd.Series(fc['yhat'].values[n_hist:n_hist + h], index=fut_idx)
    return fitted, futuros

# ── Gráfica ───────────────────────────────────────────────────────────────────

def grafica(serie, mets, titulo):
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f9f9f9')

    # Serie real con marcadores
    ax.plot(serie.index, serie.values, color='#2c3e50', lw=2,
            marker='o', markersize=4, label='Real', zorder=3)

    # Línea vertical separadora histórico / proyección
    try:
        ax.axvline(x=serie.index[-1], color='gray', linestyle='--', lw=1, alpha=0.7, label='Inicio proyección')
    except Exception:
        pass

    for k, d in mets.items():
        c = COLORES.get(k, 'gray')
        # Ajuste histórico
        ax.plot(serie.index, d['fitted'].values, color=c, ls='--', lw=2, alpha=0.85,
                label=d['label'])
        # Proyección futura
        if len(d['futuros']):
            xs = [serie.index[-1]] + list(d['futuros'].index)
            ys = [serie.iloc[-1]]  + list(d['futuros'].values)
            ax.plot(xs, ys, color=c, lw=2, marker=None)
            # Banda de confianza suave solo para Prophet
            if k == 'prophet':
                ax.fill_between(xs, [v * 0.95 for v in ys], [v * 1.05 for v in ys],
                                alpha=0.15, color=c, label='Intervalo confianza')

    ax.set_title(titulo, fontsize=13, fontweight='bold', color='#2c3e50')
    ax.set_xlabel('Fecha', fontsize=9)
    ax.set_ylabel('Ventas', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=8, loc='best')
    if isinstance(serie.index, pd.DatetimeIndex):
        import matplotlib.dates as mdates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    return fig_base64(fig)

def resumen(mets):
    def val(e, k): return f"{e[k]} %" if k == 'mape' and e[k] else (e[k] or 'N/A')
    filas = sorted([{'metodo': d['label'], 'mape': val(d['errores'], 'mape'),
                     'mse': val(d['errores'], 'mse'), 'rmse': val(d['errores'], 'rmse')}
                    for d in mets.values()],
                   key=lambda f: float(str(f['mape']).replace(' %','')) if f['mape'] != 'N/A' else 1e9)
    if filas: filas[0]['mejor'] = True
    return filas

# ── Ruta ─────────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    resultados, error_msg, metodos_sel = None, None, []
    if request.method == 'POST':
        try:
            archivo     = request.files.get('archivo')
            n           = int(request.form.get('n', 3))
            fecha_fin   = request.form.get('fecha_fin', '').strip()
            metodos_sel = request.form.getlist('metodos') or ['promedio_movil']
            if not archivo: raise ValueError("Carga un archivo CSV.")

            import io as _io
            contenido = archivo.read()
            sep = ';' if b';' in contenido.split(b'\n')[0] else ','
            df = pd.read_csv(_io.BytesIO(contenido), sep=sep)
            col0 = df.columns[0]
            try: df[col0] = pd.to_datetime(df[col0]); tiene_fechas = True
            except: tiene_fechas = False

            fecha_fin_dt = pd.to_datetime(fecha_fin) if fecha_fin else None
            resultados   = {}

            for col in df.columns[1:]:
                try:
                    temp  = pd.DataFrame({'fecha': df[col0], 'val': pd.to_numeric(df[col], errors='coerce')}).dropna()
                    serie = pd.Series(temp['val'].values,
                                      index=pd.to_datetime(temp['fecha'].values) if tiene_fechas else range(len(temp)))

                    if tiene_fechas and len(serie) >= 2:
                        freq    = pd.infer_freq(serie.index) or 'MS'
                        off     = pd.tseries.frequencies.to_offset(freq)
                        fut_idx = (pd.date_range(serie.index[-1] + off, end=fecha_fin_dt, freq=freq)
                                   if fecha_fin_dt else
                                   pd.date_range(serie.index[-1] + off, periods=6, freq=freq))
                    else:
                        fut_idx = (pd.date_range(pd.Timestamp.today(), end=fecha_fin_dt, freq='MS')
                                   if fecha_fin_dt else pd.RangeIndex(len(serie), len(serie) + 6))

                    mets = {}
                    if 'promedio_movil' in metodos_sel:
                        f, fut = pm(serie, n, fut_idx)
                        mets['promedio_movil'] = empaquetar(f'Promedio Móvil (N={n})', f, fut, errores(serie, f))

                    if 'ses' in metodos_sel:
                        f, fut, alpha = ses(serie, fut_idx)
                        mets['ses'] = empaquetar('Suavización Exponencial', f, fut, errores(serie, f), {'alpha': alpha})

                    if 'prophet' in metodos_sel:
                        try:
                            f, fut = prophet(serie, fut_idx)
                            mets['prophet'] = empaquetar('Prophet (Meta)', f, fut, errores(serie, f))
                        except Exception as ep:
                            mets['prophet'] = empaquetar('Prophet (error)',
                                pd.Series(dtype=float), pd.Series(dtype=float),
                                {'mape': None, 'mse': None, 'rmse': None},
                                {'error': str(ep)})

                    resultados[col] = {'metodos': mets,
                                       'grafica': grafica(serie, mets, f'Pronóstico – {col}'),
                                       'resumen': resumen(mets)}
                except Exception as ecol:
                    resultados[col] = {'metodos': {}, 'grafica': None,
                                       'resumen': [], 'error_col': str(ecol)}
        except Exception as exc:
            error_msg = str(exc)

    return render_template('optimizacion.html', resultados=resultados, error_msg=error_msg,
                           metodos_sel=metodos_sel, prophet_disponible=PROPHET_OK)

if __name__ == '__main__':
    app.run(debug=True)