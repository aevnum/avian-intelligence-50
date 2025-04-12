# app.py
import os
import json
import uuid
import logging
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import openai # Import OpenAI library
import cv2 # OpenCV is often a dependency for Ultralytics or image manipulation

# --- Basic Setup & Configuration ---
logging.basicConfig(level=logging.INFO) # Basic logging
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = os.path.join('static', 'results') # Store YOLO output images here
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # 16MB upload limit
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# --- Ensure Folders Exist ---
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)
os.makedirs(os.path.join('static', 'bird_images'), exist_ok=True)
os.makedirs(os.path.join('static', 'bird_calls'), exist_ok=True)

# --- Load Bird Data ---
def load_bird_data(filepath='data/birds.json'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f: # Specify encoding
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

# --- Configure OpenAI API ---
# Load API key from environment variable for security
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    logging.warning("OPENAI_API_KEY environment variable not set. Chat feature will be disabled.")
    openai_client = None
else:
    # It's good practice to create the client instance once
    try:
        openai_client = openai.OpenAI(api_key=openai_api_key)
        logging.info("OpenAI client configured.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenAI client: {e}")
        openai_client = None


# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Prediction Function ---
def predict_birds(image_path):
    if not model:
        return None, "Model not loaded", []

    try:
        # Run YOLOv8 inference
        results = model(image_path, verbose=False) # verbose=False to reduce console output

        # Check if results are available
        if not results or not results[0]:
             logging.warning(f"No results returned by YOLO model for {image_path}")
             return None, "Model did not return results", []

        processed_results = results[0] # Assuming results is a list

        # Save the image with bounding boxes drawn
        output_filename = f"{uuid.uuid4()}.jpg"
        output_path = os.path.join(app.config['RESULT_FOLDER'], output_filename)
        processed_results.save(filename=output_path) # Use save method
        logging.info(f"Saved prediction result image to {output_path}")

        # Extract class names and confidences
        detected_classes = []
        names = processed_results.names # Dictionary {index: 'class_name'}
        conf_threshold = 0.4 # Example confidence threshold - adjust as needed!

        for box in processed_results.boxes:
            try:
                class_index = int(box.cls[0])
                class_name = names[class_index]
                confidence = float(box.conf[0])

                if confidence >= conf_threshold:
                    # Add unique classes only
                    if class_name not in [d['name'] for d in detected_classes]:
                        detected_classes.append({'name': class_name, 'confidence': round(confidence, 2)})
            except (IndexError, KeyError, ValueError) as e:
                logging.warning(f"Could not parse box data: {box}. Error: {e}")
                continue # Skip this problematic box

        # Sort detections by confidence (optional)
        detected_classes.sort(key=lambda x: x['confidence'], reverse=True)

        # Return the path relative to the 'static' folder for use with url_for
        relative_output_path = f"results/{output_filename}"
        return relative_output_path, None, detected_classes

    except Exception as e:
        logging.exception(f"Error during prediction for {image_path}") # Log full traceback
        return None, f"Prediction error: {e}", []

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static_file(path):
    """Serves static files (CSS, JS, images, audio)."""
    # Added for explicit static file serving if needed, though url_for usually handles it.
    # Ensure the path doesn't try to escape the static directory (basic security)
    safe_path = secure_filename(path)
    return send_from_directory('static', safe_path)

@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Handles image upload and returns prediction results."""
    if 'photo' not in request.files:
        return jsonify({'error': 'No photo part in the request'}), 400
    file = request.files['photo']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Use a unique name for the temporary file to avoid conflicts
        temp_filename = f"{uuid.uuid4()}_{filename}"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)

        try:
            file.save(temp_filepath)
            logging.info(f"Uploaded file saved temporarily to {temp_filepath}")

            # Run prediction
            result_image_rel_path, error_msg, detected_classes = predict_birds(temp_filepath)

            # **** Add Logging Here ****
            logging.info(f"predict_birds returned relative path: {result_image_rel_path}")
            # **** End Logging ****

            if error_msg:
                # **** Optional: Log error before returning ****
                logging.error(f"Prediction error message: {error_msg}")
                return jsonify({'error': error_msg}), 500

            # Check if path is valid before generating URL
            if result_image_rel_path:
                result_image_url = url_for('static', filename=result_image_rel_path, _external=False)
                # **** Add Logging Here ****
                logging.info(f"Generated result_image_url: {result_image_url}")
                # **** End Logging ****
            else:
                result_image_url = None # Handle case where no image path was returned
                logging.warning("No result_image_rel_path returned, URL will be null.")


            response_data = {
                'result_image_url': result_image_url,
                'detections': detected_classes
            }
            # **** Add Logging Here ****
            logging.info(f"Returning JSON: {response_data}")
            # **** End Logging ****
            return jsonify(response_data)

        except Exception as e:
             logging.exception("Error handling prediction request")
             return jsonify({'error': 'Failed to process image'}), 500
        finally:
            # Clean up uploaded file
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                    logging.info(f"Removed temporary file: {temp_filepath}")
                except OSError as e:
                    logging.error(f"Error removing temporary file {temp_filepath}: {e}")
    else:
         return jsonify({'error': 'Invalid file type'}), 400

@app.route('/bird_info/<path:bird_name>') # Use path converter to handle names with slashes if any
def get_bird_info(bird_name):
    """Serves detailed information about a specific bird species."""
    # Sanitize bird_name just in case, although less critical if just used as dict key
    safe_bird_name = secure_filename(bird_name) # Basic check
    info = bird_data.get(safe_bird_name) # Use sanitized name

    if info:
        # Create a copy to modify paths without altering original data
        info_copy = info.copy()
        # Generate URLs for static assets if paths are present
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
    """Handles chat requests using OpenAI API."""
    if not openai_client: # Check if OpenAI client is configured
        return jsonify({'reply': "Sorry, the chat feature is not configured or the API key is missing."}), 503 # Service Unavailable

    data = request.get_json()
    if not data or 'bird_name' not in data or 'message' not in data:
        return jsonify({'error': 'Missing bird_name or message in request'}), 400

    bird_name = data['bird_name']
    user_message = data['message']
    chat_history = data.get('history', []) # Optional: Get chat history for context

    # Fetch bird context from our data
    bird_details = bird_data.get(bird_name, {})
    context_summary = f"Genus: {bird_details.get('genus', 'N/A')}, Locations: {bird_details.get('locations', 'N/A')}, Info: {bird_details.get('short_info', 'N/A')}."

    # Construct messages for OpenAI API (using Chat Completions format)
    messages = [
        {"role": "system", "content": f"You are a helpful ornithology assistant specializing in bird information. The user is asking about the '{bird_name}'. Basic info: {context_summary}. Keep answers concise and relevant to birds."},
    ]
    # Add previous conversation history (optional, but improves context)
    for entry in chat_history[-4:]: # Limit history length
        messages.append({"role": entry["role"], "content": entry["content"]})

    # Add the current user message
    messages.append({"role": "user", "content": user_message})

    try:
        logging.info(f"Sending request to OpenAI for bird: {bird_name}")
        # Use the Chat Completions endpoint
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if you have access and budget
            messages=messages,
            max_tokens=150,  # Adjust as needed
            temperature=0.7 # Adjust for creativity vs factualness
        )

        # Extract the reply
        ai_reply = response.choices[0].message.content.strip()
        logging.info(f"Received reply from OpenAI for bird: {bird_name}")

        return jsonify({'reply': ai_reply})

    except openai.APIError as e:
        logging.error(f"OpenAI API returned an API Error: {e}")
        return jsonify({'reply': f"Sorry, I encountered an error communicating with the AI assistant (API Error: {e})."}), 503
    except openai.AuthenticationError as e:
        logging.error(f"OpenAI Authentication Error: {e}")
        return jsonify({'reply': "Sorry, there's an issue with the AI assistant configuration (Authentication Error)."}), 500
    except openai.RateLimitError as e:
         logging.error(f"OpenAI Rate Limit Error: {e}")
         return jsonify({'reply': "Sorry, the AI assistant is currently busy. Please try again shortly."}), 429
    except Exception as e:
        logging.exception("Unexpected error in chat handler")
        return jsonify({'reply': "Sorry, an unexpected error occurred while contacting the AI assistant."}), 500


# --- Main Execution ---
if __name__ == '__main__':
    # Use host='0.0.0.0' to make it accessible on your network (use with caution)
    # Use debug=True for development ONLY, it's a security risk in production
    app.run(debug=True, host='127.0.0.1', port=5000)