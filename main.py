from flask import Flask, request, jsonify, send_from_directory
import joblib
import pandas as pd
import os
import sys
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Ruta al archivo del modelo
modelo_path = os.path.join(os.getcwd(), 'modelo_fraudev2.pkl')

# Cargar el modelo si existe
if os.path.exists(modelo_path):
    modelo = joblib.load(modelo_path)
else:
    print("Archivo de modelo no encontrado")
    sys.exit(1)

# Inicializar el escalador
escalador = StandardScaler()

# Definir las columnas necesarias
columnas_modelo = [
    'income', 'name_email_similarity', 'current_address_months_count', 
    'customer_age', 'days_since_request', 'bank_branch_count_8w', 
    'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'email_is_free', 
    'phone_home_valid', 'phone_mobile_valid', 'bank_months_count', 
    'has_other_cards', 'proposed_credit_limit', 'foreign_request', 
    'session_length_in_minutes', 'keep_alive_session', 
    'device_distinct_emails_8w', 'payment_type_AA', 'payment_type_AB', 
    'payment_type_AC', 'payment_type_AD', 'payment_type_AE', 
    'employment_status_CA', 'employment_status_CB', 'employment_status_CC', 
    'employment_status_CD', 'employment_status_CE', 'employment_status_CF', 
    'employment_status_CG', 'housing_status_BA', 'housing_status_BB', 
    'housing_status_BC', 'housing_status_BD', 'housing_status_BE', 
    'housing_status_BF', 'housing_status_BG', 'source_INTERNET', 
    'source_TELEAPP', 'device_os_linux', 'device_os_macintosh', 
    'device_os_other', 'device_os_windows', 'device_os_x11'
]

def preparar_dataframe(df):
    # Verificar si las columnas requeridas están en el DataFrame
    faltantes = [col for col in columnas_modelo if col not in df.columns]
    if faltantes:
        return None, f"Faltan las siguientes columnas: {', '.join(faltantes)}"
    
    df_filtrado = df[columnas_modelo].copy()
    return df_filtrado, None

def preprocesar_datos(data_df):
    data_df_normalizado = escalador.fit_transform(data_df)
    return pd.DataFrame(data_df_normalizado, columns=data_df.columns)

@app.route('/')
def home():
    return '''
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predicción de Fraude</title>
    <style>
        body {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
            text-align: center;
            background-color: rgb(13, 14, 26);
            position: relative;
            color: white;
        }
        .circle {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border-radius: 50%;
            opacity: 0.7;
            z-index: 0;
        }
        .circle-large {
            width: 450px;
            height: 450px;
            background-color: rgb(30, 11, 84);
            filter: blur(50px);
        }
        .circle-purple {
            width: 300px;
            height: 300px;
            background-color: rgb(25, 52, 150);
            filter: blur(50px);
        }
        .circle-blue {
            width: 225px;
            height: 225px;
            background-color: rgb(25, 52, 150);
            filter: blur(40px);
        }
        .circle-pink {
            width: 112.5px;
            height: 112.5px;
            background-color: rgb(168, 0, 30);
            filter: blur(20px);
        }
        form {
            margin-top: 20px;
            position: relative;
            z-index: 2;
        }
        input[type="file"] {
            display: none;
        }
        .file-label {
            border: 2px solid white;
            padding: 10px;
            border-radius: 5px;
            background-color: transparent;
            color: white;
            cursor: pointer;
            z-index: 2;
        }
        .selected-file {
            margin: 20px 0;
            color: white;
            z-index: 1;
        }
        h1, p {
            z-index: 1;
            position: relative;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
        }
        a {
            color: white;
            font-size: 22px;
            text-decoration: underline;
            cursor: pointer;
            z-index: 3;
            display: inline-block;
        }
        #loading {
            display: none;
            margin-top: 20px;
            font-size: 24px;
            font-weight: bold;
            color: white;
        }
        #result {
            color: white;
            font-size: 22px;
            margin-top: 20px;
            z-index: 1;
        }
        .status-container {
            min-height: 50px;
            margin-top: 20px;
            z-index: 1;
        }
    </style>
</head>
<body>
    <div class="circle circle-large"></div>
    <div class="circle circle-purple"></div>
    <div class="circle circle-blue"></div>
    <div class="circle circle-pink"></div>

    <h1>Subir archivo para predicción de fraude</h1>
    <form id="uploadForm" enctype="multipart/form-data">
        <label class="file-label" for="file-input">Seleccionar archivo</label>
        <input type="file" id="file-input" name="file" required onchange="updateFileName()">
        <div class="selected-file" id="selected-file">Archivos seleccionados: Ninguno</div>
        <button type="submit">Enviar</button>
    </form>
    <p>Recordar que debe tener las columnas adecuadas.</p>
    <div class="status-container">
        <div id="loading">Cargando<span id="dots">...</span></div>
        <div id="result"></div>
    </div>

    <script>
        function updateFileName() {
            const input = document.getElementById('file-input');
            const label = document.getElementById('selected-file');
            const fileName = input.files.length > 0 ? input.files[0].name : 'Ninguno';
            label.textContent = `Archivos seleccionados: ${fileName}`;
        }

        document.getElementById('uploadForm').onsubmit = async function(event) {
            event.preventDefault();
            const formData = new FormData(this);
            const loadingMessage = document.getElementById('loading');
            const dots = document.getElementById('dots');

            loadingMessage.style.display = 'block';

            let dotCount = 0;
            const dotAnimation = setInterval(() => {
                dotCount = (dotCount + 1) % 4;
                dots.textContent = '.'.repeat(dotCount);
            }, 500);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();

                clearInterval(dotAnimation);
                loadingMessage.style.display = 'none';

                if (data.output_file) {
                    const downloadLink = `<a href="/download/${data.output_file}" download>Descargar resultados</a>`;
                    document.getElementById('result').innerHTML = downloadLink;
                } else {
                    document.getElementById('result').innerText = data.error;
                }
            } catch (error) {
                clearInterval(dotAnimation);
                loadingMessage.style.display = 'none';
                document.getElementById('result').innerText = 'Error en la conexión. Inténtalo de nuevo.';
                console.error(error);
            }
        };
    </script>
</body>
</html>

    '''

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No hay parte de archivo en la solicitud'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Archivo no cargado'}), 400
    
    try:
        data_df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'Error al leer el archivo CSV: {str(e)}'}), 400

    data_df_preparado, error = preparar_dataframe(data_df)
    if error:
        return jsonify({'error': error}), 400

    data_df_preprocesado = preprocesar_datos(data_df_preparado)

    prediccion = modelo.predict(data_df_preprocesado)
    resultados_df = pd.DataFrame(data={'predicciones': prediccion})

    output_folder = 'resultados'
    os.makedirs(output_folder, exist_ok=True)

    original_filename = os.path.splitext(file.filename)[0]
    output_file_name = f"resultado_{original_filename}.csv"
    output_file_path = os.path.join(output_folder, output_file_name)

    resultados_df.to_csv(output_file_path, index=False)

    return jsonify({'message': 'Predicciones guardadas en CSV', 'output_file': output_file_name})

@app.route('/download/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('resultados', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=os.getenv('PORT',default=5000))