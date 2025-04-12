import os
import json
import uuid
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import google.generativeai as genai  # Import Gemini API
import cv2

# --- Basic Setup & Configuration ---
logging.basicConfig(level=logging.INFO)
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = os.path.join('static', 'results')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# --- Ensure Folders Exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'bird_images'), exist_ok=True)
os.makedirs(os.path.join('static', 'bird_calls'), exist_ok=True)

# --- Load Bird Data ---
def load_bird_data(filepath='data/birds.json'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: Bird data file not found at {filepath}")
        return {}
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {filepath}")
        return {}

bird_data = load_bird_data()
if not bird_data:
    logging.warning("Bird data is empty. Features relying on it might not work.")

# --- Load YOLOv8 Model ---
try:
    model = YOLO('model/best.pt')
    logging.info("YOLOv8 model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading YOLOv8 model: {e}")
    model = None

# --- Configure Gemini API ---
try:
    with open('apifree.txt', 'r') as f:
        gemini_api_key = f.read().strip()
    logging.info("Gemini API key loaded from file.")
    genai.configure(api_key=gemini_api_key)
    model_gemini = genai.GenerativeModel("gemini-2.0-flash")  # Initialize Gemini 2.0 Flash model
    logging.info("Gemini client configured.")

except FileNotFoundError:
    logging.warning("gemini_api_key.txt not found. Chat feature will be disabled.")
    model_gemini = None

except Exception as e:
    logging.error(f"Failed to initialize Gemini client: {e}")
    model_gemini = None

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Prediction Function ---
def predict_birds(image_path):
    if not model:
        return None, "Model not loaded", []

    try:
        results = model(image_path, verbose=False)

        if not results or not results[0]:
            logging.warning(f"No results returned by YOLO model for {image_path}")
            return None, "Model did not return results", []

        processed_results = results[0]

        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        processed_results.save(filename=output_path)
        logging.info(f"Saved prediction result image to {output_path}")

        detected_classes = []
        names = processed_results.names
        conf_threshold = 0.4

        for box in processed_results.boxes:
            try:
                class_index = int(box.cls[0])
                class_name = names[class_index]
                confidence = float(box.conf[0])

                if confidence >= conf_threshold:
                    if class_name not in [d['name'] for d in detected_classes]:
                        detected_classes.append({'name': class_name, 'confidence': round(confidence, 2)})
            except (IndexError, KeyError, ValueError) as e:
                logging.warning(f"Could not parse box data: {box}. Error: {e}")
                continue

        detected_classes.sort(key=lambda x: x['confidence'], reverse=True)

        relative_output_path = f"results/{output_filename}"
        return relative_output_path, None, detected_classes

    except Exception as e:
        logging.exception(f"Error during prediction for {image_path}")
        return None, f"Prediction error: {e}", []

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static_file(path):
    safe_path = secure_filename(path)
    return send_from_directory('static', safe_path)

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo part in the request'}), 400
    file = request.files['photo']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        temp_filename = f"{uuid.uuid4()}_{filename}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        try:
            file.save(temp_filepath)
            logging.info(f"Uploaded file saved temporarily to {temp_filepath}")

            result_image_rel_path, error_msg, detected_classes = predict_birds(temp_filepath)

            logging.info(f"predict_birds returned relative path: {result_image_rel_path}")

            if error_msg:
                logging.error(f"Prediction error message: {error_msg}")
                return jsonify({'error': error_msg}), 500

            if result_image_rel_path:
                result_image_url = url_for('static', filename=result_image_rel_path, _external=False)
                logging.info(f"Generated result_image_url: {result_image_url}")
            else:
                result_image_url = None
                logging.warning("No result_image_rel_path returned, URL will be null.")

            response_data = {
                'result_image_url': result_image_url,
                'detections': detected_classes
            }
            logging.info(f"Returning JSON: {response_data}")
            return jsonify(response_data)

        except Exception as e:
            logging.exception("Error handling prediction request")
            return jsonify({'error': 'Failed to process image'}), 500
        finally:
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                    logging.info(f"Removed temporary file: {temp_filepath}")
                except OSError as e:
                    logging.error(f"Error removing temporary file {temp_filepath}: {e}")
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/bird_info/<path:bird_name>')
def get_bird_info(bird_name):
    safe_bird_name = secure_filename(bird_name)
    info = bird_data.get(safe_bird_name)

    if info:
        info_copy = info.copy()
        if info_copy.get('image_path'):
            info_copy['image_path'] = url_for('static', filename=info_copy['image_path'].replace('static/', '', 1), _external=False)
        if info_copy.get('audio_path'):
            info_copy['audio_path'] = url_for('static', filename=info_copy['audio_path'].replace('static/', '', 1), _external=False)
        return jsonify(info_copy)
    else:
        logging.warning(f"Bird info requested for '{safe_bird_name}', but not found in data.")
        return jsonify({'error': 'Bird species not found in database'}), 404

@app.route('/chat', methods=['POST'])
def handle_chat():
    if not model_gemini:
        return jsonify({'reply': "Sorry, the chat feature is not configured or the API key is missing."}), 503

    data = request.get_json()
    if not data or 'bird_name' not in data or 'message' not in data:
        return jsonify({'error': 'Missing bird_name or message in request'}), 400

    bird_name = data['bird_name']
    user_message = data['message']
    chat_history = data.get('history', [])

    bird_details = bird_data.get(bird_name, {})
    context_summary = f"Genus: {bird_details.get('genus', 'N/A')}, Locations: {bird_details.get('locations', 'N/A')}, Info: {bird_details.get('short_info', 'N/A')}."

    messages = [
        {"role": "system", "content": f"You are a helpful ornithology assistant specializing in bird information. The user is asking about the '{bird_name}'. Basic info: {context_summary}. Keep answers concise and relevant to birds."},
    ]
    for entry in chat_history[-4:]:
        messages.append({"role": entry["role"], "content": entry["content"]})

    messages.append({"role": "user", "content": user_message})

    try:
        logging.info(f"Sending request to Gemini for bird: {bird_name}")
        # Gemini API interaction
        prompt = "\n".join([msg["content"] for msg in messages]) #convert messages to one string.
        response = model_gemini.generate_content(prompt)
        ai_reply = response.text.strip()
        logging.info(f"Received reply from Gemini for bird: {bird_name}")

        return jsonify({'reply': ai_reply})

    except Exception as e:
        logging.exception("Unexpected error in chat handler")
        return jsonify({'reply': "Sorry, an unexpected error occurred while contacting the AI assistant."}), 500

# --- Main Execution ---
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)