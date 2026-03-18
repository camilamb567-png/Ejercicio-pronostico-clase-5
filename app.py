from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__)

# PROMEDIO MÓVIL 
def promedio_movil(data, n):
    return data.shift(1).rolling(window=n).mean()

# CÁLCULO DE ERRORES 
def calcular_error(real, pronostico):
    df = pd.DataFrame({
        'real': real,
        'pronostico': pronostico
    }).dropna()

    error = df['real'] - df['pronostico']

    # MAPE
    mape = (abs(error / df['real']).mean()) * 100

    # MAPE' 
    mape_prima = (abs(error / df['real']).mean()) * 100

    # MSE
    mse = (error ** 2).mean()

    # RMSE
    rmse = np.sqrt(mse)

    return mape, mape_prima, mse, rmse

# RUTA PRINCIPAL
@app.route('/', methods=['GET', 'POST'])
def index():
    resultados = None

    if request.method == 'POST':
        archivo = request.files['archivo']
        n = int(request.form['n'])

        if archivo:
            df = pd.read_csv(archivo, sep=';')

            resultados = {}

            for columna in df.columns[1:]:
                serie = pd.to_numeric(df[columna], errors='coerce')

                pronostico = promedio_movil(serie, n)

                mape, mape_prima, mse, rmse = calcular_error(serie, pronostico)

                resultados[columna] = {
                    'pronostico': [
                        round(x, 2) if pd.notna(x) else None
                        for x in pronostico
                    ],
                    'mape': round(mape, 2),
                    'mape_prima': round(mape_prima, 2),
                    'mse': round(mse, 2),
                    'rmse': round(rmse, 2)
                }

    return render_template('optimizacion.html', resultados=resultados)

# EJECUTAR APP
if __name__ == '__main__':
    app.run(debug=True)