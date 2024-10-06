from flask import Flask, request, jsonify, render_template
import os
import whisper
import logging
import traceback
from flask_cors import CORS
import time
import json

app = Flask(__name__)
CORS(app)  # Habilitar CORS para permitir peticiones desde el navegador

# Set up logging to trace issues
def setup_logging():
    logging.basicConfig(filename='app.log', level=logging.INFO,  # Cambiar a INFO para reducir la verbosidad
                        format='%(asctime)s %(levelname)s %(message)s',
                        filemode='w')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Mostrar mensajes importantes en la consola
    logging.getLogger().addHandler(console_handler)

setup_logging()

# Load Whisper model
try:
    logging.info("Cargando el modelo Whisper...")
    model = whisper.load_model("medium", device="cpu")  # Usar modelo "medium" y forzar el uso de CPU
    logging.info("Modelo cargado exitosamente.")
except Exception as e:
    logging.error(f"Error al cargar el modelo: {e}")
    model = None

@app.route('/')
def index():
    logging.info("Renderizando la página principal.")
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    logging.info("Solicitud de transcripción recibida.")
    if 'file' not in request.files:
        logging.warning("No se encontró el archivo en la solicitud.")
        return jsonify({'error': 'No se encontró el archivo'}), 400
    file = request.files['file']
    if file.filename == '':
        logging.warning("El nombre del archivo está vacío.")
        return jsonify({'error': 'Nombre de archivo vacío'}), 400
    if file and model:
        try:
            timestamp = int(time.time())
            filename = f"{timestamp}_{file.filename}"
            filepath = os.path.join('uploads', filename)
            logging.info(f"Guardando archivo en la ruta: {filepath}")
            file.save(filepath)
            # Verificar si el archivo fue guardado correctamente
            if not os.path.exists(filepath):
                logging.error("El archivo no se guardó correctamente en la ruta especificada.")
                return jsonify({'error': 'Error al guardar el archivo'}), 500
            # Run Whisper model transcription using Python API
            logging.info("Ejecutando la transcripción usando el modelo Whisper en Python...")
            result = model.transcribe(filepath, language='ja', word_timestamps=True)  # Transcribir en japonés con timestamps detallados
            logging.info("Transcripción completada.")
            # Guardar el resultado de la transcripción en un archivo con el mismo nombre que el mp3 pero extensión .json
            lyrics_filepath = os.path.splitext(filepath)[0] + ".json"
            with open(lyrics_filepath, 'w') as f:
                json.dump(result, f)
            logging.info(f"Archivo de transcripción guardado en la ruta: {lyrics_filepath}")
            os.remove(filepath)
            logging.info("Archivo de audio transcrito eliminado exitosamente.")
            return jsonify(result), 200
        except Exception as e:
            logging.error(f"Error al transcribir el archivo: {str(e)}")
            logging.error(traceback.format_exc())
            return jsonify({'error': f'Error al transcribir el archivo: {str(e)}'}), 500
    else:
        logging.error("Error al cargar el archivo o el modelo no está disponible.")
        return jsonify({'error': 'Error al cargar el archivo o el modelo no está disponible'}), 400

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    logging.info("Iniciando la aplicación Flask...")
    app.run(debug=True, threaded=False, use_reloader=False)