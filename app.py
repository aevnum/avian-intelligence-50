import os
import json
import uuid
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import google.generativeai as genai  # Import Gemini API
import cv2
from huggingface_hub import hf_hub_download

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
# --- Download and Load YOLOv8 Model ---
model = None # Initialize as None
try:
    # Define Hugging Face repo details - CHANGE THESE
    HF_REPO_ID = "aevnum/avian-intelligence-yolo" # <<<--- YOUR HF REPO ID
    HF_FILENAME = "best.pt"
    MODEL_CACHE_DIR = "model_cache" # Can be any directory name

    # Ensure the local model cache directory exists
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    logging.info(f"Downloading model {HF_FILENAME} from {HF_REPO_ID}...")
    # Use HF Token from environment variable for download
    hf_token = os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if not hf_token:
         logging.warning("HUGGING_FACE_HUB_TOKEN not set. Download might fail for private repos or hit rate limits.")

    downloaded_model_path = hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=HF_FILENAME,
        cache_dir=MODEL_CACHE_DIR,
        force_filename=HF_FILENAME, # Helps ensure consistent naming if cache is used
        token=hf_token
    )
    logging.info(f"Model downloaded to: {downloaded_model_path}")

    # Load the downloaded model
    model = YOLO(downloaded_model_path)
    logging.info("YOLOv8 model loaded successfully from downloaded file.")

except Exception as e:
    logging.exception("Error downloading or loading YOLOv8 model from Hugging Face Hub") # Log full traceback
    model = None

# --- Configure Gemini API ---
model_gemini = None # Initialize as None
try:
    gemini_api_key = os.environ.get('GEMINI_API_KEY') # Get key from environment
    if not gemini_api_key:
        logging.warning("GEMINI_API_KEY environment variable not set. Chat feature will be disabled.")
    else:
        logging.info("Configuring Gemini API...")
        genai.configure(api_key=gemini_api_key)
        # Consider making model name configurable too via env var if needed
        # GEMINI_MODEL = os.environ.get('GEMINI_MODEL_NAME', 'gemini-1.5-flash-latest')
        model_gemini = genai.GenerativeModel('gemini-1.5-flash-latest') # Or use variable
        logging.info(f"Gemini client configured with model.")

except Exception as e:
    logging.exception("Failed to initialize Gemini client") # Log full traceback
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
    # Use host='0.0.0.0' to be accessible within the container
    # Port is usually set by the deployment platform via PORT env var
    port = int(os.environ.get('PORT', 5000)) # Default to 5000 if PORT not set
    app.run(host='0.0.0.0', port=port)