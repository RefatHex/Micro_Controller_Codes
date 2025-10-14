from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
from pyngrok import ngrok
import os
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
import tempfile
from PIL import Image
from datetime import datetime
import requests
import json
import pandas as pd
import openpyxl

# ML integration -----------------------------------------------------------
# Import ML helpers from ML.py. This allows app.py to call ML.predict(...) without
# retraining; artifacts are loaded from ml_artifacts.joblib by load_artifacts().
from ML import load_artifacts, predict

# Load model artifacts at startup (best-effort). If artifacts are missing the app
# will still start; run `python ML.py` once to create artifacts (ml_artifacts.joblib).
try:
    load_artifacts()
except Exception:
    # Artifacts not present yet — that's okay for now. They can be created by
    # running ML.py manually. We avoid failing fast so development can continue.
    pass
# -------------------------------------------------------------------------

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"])

# Initialize Roboflow client
client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

# Create photos folder if it doesn't exist
PHOTOS_FOLDER = os.path.join(os.path.dirname(__file__), 'photos')
if not os.path.exists(PHOTOS_FOLDER):
    os.makedirs(PHOTOS_FOLDER)
    print(f"Created photos folder: {PHOTOS_FOLDER}")

# Excel file path for tracking
EXCEL_FILE = os.path.join(os.path.dirname(__file__), 'image_location_log.xlsx')

def get_location_from_wifi(wifi_data):
    """Get location using Google Geolocation API with WiFi data"""
    try:
        api_key = os.getenv("GOOGLE_GEOLOCATION_API_KEY")
        if not api_key:
            return None, None, "No Google API key"
        
        url = f"https://www.googleapis.com/geolocation/v1/geolocate?key={api_key}"
        
        # Parse WiFi data from ESP32
        wifi_json = json.loads(wifi_data)
        
        response = requests.post(url, json=wifi_json, timeout=10)
        if response.status_code == 200:
            location_data = response.json()
            lat = location_data.get('location', {}).get('lat')
            lng = location_data.get('location', {}).get('lng')
            accuracy = location_data.get('accuracy', 'unknown')
            return lat, lng, f"accuracy: {accuracy}m"
        else:
            return None, None, f"API error: {response.status_code}"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def log_to_excel(image_name, latitude, longitude, location_info, trash_detected, trash_type, timestamp):
    """Log image information to Excel file"""
    try:
        # Create new row data
        new_row = {
            'Timestamp': timestamp,
            'Image_Name': image_name,
            'Latitude': latitude if latitude else 'N/A',
            'Longitude': longitude if longitude else 'N/A',
            'Location_Info': location_info,
            'Trash_Detected': trash_detected,
            'Trash_Type': trash_type
        }
        
        # Check if Excel file exists
        if os.path.exists(EXCEL_FILE):
            # Read existing data
            df = pd.read_excel(EXCEL_FILE)
            # Append new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            # Create new DataFrame
            df = pd.DataFrame([new_row])
        
        # Save to Excel
        df.to_excel(EXCEL_FILE, index=False)
        print(f"Logged to Excel: {image_name}")
        
    except Exception as e:
        print(f"Error logging to Excel: {str(e)}")

def rgb565_to_jpeg(rgb565_data, width=320, height=240):
    """Convert RGB565 data to JPEG format - QVGA resolution"""
    # Convert RGB565 bytes to numpy array
    rgb565_array = np.frombuffer(rgb565_data, dtype=np.uint16)
    
    # Convert RGB565 to RGB888
    r = ((rgb565_array & 0xF800) >> 11) << 3
    g = ((rgb565_array & 0x07E0) >> 5) << 2
    b = (rgb565_array & 0x001F) << 3
    
    # Create RGB array
    rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
    rgb_array[:, :, 0] = r.reshape(height, width)
    rgb_array[:, :, 1] = g.reshape(height, width)
    rgb_array[:, :, 2] = b.reshape(height, width)
    
    # Convert to PIL Image and save as JPEG
    image = Image.fromarray(rgb_array, 'RGB')
    return image

def load_or_create_excel(file_name):
    if os.path.exists(file_name):
        workbook = openpyxl.load_workbook(file_name)
    else:
        workbook = openpyxl.Workbook()
        workbook.active.append(["Timestamp", "Ph_Value", "Turbidity", "Temperature", "Flow_Value"])  # Header
        workbook.save(file_name)
    return workbook

@app.route('/check', methods=['POST', 'OPTIONS'])
def check():
    # Handle preflight OPTIONS request
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response
    
    try:
        # Get WiFi data for location if available
        wifi_data = request.form.get('wifi_data', None) if request.form else None
        latitude, longitude, location_info = None, None, "No location data"
        
        if wifi_data:
            latitude, longitude, location_info = get_location_from_wifi(wifi_data)
            print(f"Location: {latitude}, {longitude} ({location_info})")
        
        # Check if raw JPEG data was uploaded
        if request.content_type and ('application/octet-stream' in request.content_type or 'image/jpeg' in request.content_type):
            # Handle direct JPEG data from ESP32
            jpeg_data = request.get_data()
            if len(jpeg_data) == 0:
                return jsonify({'error': 'No data received'}), 400
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"esp32_photo_{timestamp}.jpg"
            saved_photo_path = os.path.join(PHOTOS_FOLDER, image_name)
            
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file_path = temp_file.name
            
            try:
                # Write JPEG data directly to file
                with open(temp_file_path, 'wb') as f:
                    f.write(jpeg_data)
                
                # Save a copy to photos folder
                with open(saved_photo_path, 'wb') as f:
                    f.write(jpeg_data)
                print(f"Photo saved: {saved_photo_path}")
                
                # Run inference
                result = client.run_workflow(
                    workspace_name="microshets",
                    workflow_id="detect-and-classify-2",
                    images={
                        "image": temp_file_path
                    },
                    use_cache=True
                )
                
                # Extract trash detection results
                trash_detected = False
                trash_name = "none"
                
                if result and len(result) > 0:
                    detection_predictions = result[0].get('detection_predictions', {})
                    predictions = detection_predictions.get('predictions', [])
                    
                    if predictions and len(predictions) > 0:
                        trash_detected = True
                        trash_name = predictions[0].get('class', 'unknown')
                
                # Log to Excel
                log_to_excel(image_name, latitude, longitude, location_info, 
                           trash_detected, trash_name, datetime.now())
                
            finally:
                # Clean up temporary file with error handling
                try:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                except PermissionError:
                    pass  # File might still be in use, skip deletion
            
            # Return simplified results with CORS headers
            response = jsonify({
                'trash_detected': trash_detected,
                'result': trash_name,
                'location': {
                    'latitude': latitude,
                    'longitude': longitude,
                    'info': location_info
                }
            })
            print(f"Trash detected: {trash_detected}, Result: {trash_name}")
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
            
        # Handle multipart form data (for testing with Postman or ESP32)
        elif 'photo' in request.files:
            # Check if a file was uploaded
            if 'photo' not in request.files:
                return jsonify({'error': 'No photo uploaded'}), 400
            
            photo = request.files['photo']
            if photo.filename == '':
                return jsonify({'error': 'No photo selected'}), 400
            
            # Generate timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_name = f"uploaded_photo_{timestamp}.jpg"
            saved_photo_path = os.path.join(PHOTOS_FOLDER, image_name)
            
            # Save uploaded file temporarily
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            temp_file_path = temp_file.name
            
            try:
                # Check if file is RGB565 format
                if photo.filename.endswith('.rgb') or photo.content_type == 'application/octet-stream':
                    # Handle RGB565 data from ESP32
                    rgb565_data = photo.read()
                    image = rgb565_to_jpeg(rgb565_data)
                    image.save(temp_file_path, 'JPEG')
                    image.save(saved_photo_path, 'JPEG')
                else:
                    # Handle regular image files
                    photo.save(temp_file_path)
                    photo.seek(0)  # Reset file pointer
                    photo.save(saved_photo_path)
                
                print(f"Photo saved: {saved_photo_path}")
                temp_file.close()  # Close file before using it
                
                # Run inference
                result = client.run_workflow(
                    workspace_name="microshets",
                    workflow_id="detect-and-classify-2",
                    images={
                        "image": temp_file_path
                    },
                    use_cache=True
                )
                
                # Extract trash detection results
                trash_detected = False
                trash_name = "none"
                
                if result and len(result) > 0:
                    detection_predictions = result[0].get('detection_predictions', {})
                    predictions = detection_predictions.get('predictions', [])
                    
                    if predictions and len(predictions) > 0:
                        trash_detected = True
                        trash_name = predictions[0].get('class', 'unknown')
                
                # Log to Excel
                log_to_excel(image_name, latitude, longitude, location_info, 
                           trash_detected, trash_name, datetime.now())
                
            finally:
                # Clean up temporary file with error handling
                try:
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
                except PermissionError:
                    pass  # File might still be in use, skip deletion
            
            # Return simplified results with CORS headers
            response = jsonify({
                'trash_detected': trash_detected,
                'result': trash_name
            })
            response.headers.add('Access-Control-Allow-Origin', '*')
            return response
        
        else:
            return jsonify({'error': 'No photo uploaded'}), 400
            
    except Exception as e:
        error_response = jsonify({'error': str(e)})
        error_response.headers.add('Access-Control-Allow-Origin', '*')
        return error_response, 500

# API route to handle POST requests and save data to Excel
@app.route('/add_to_excel', methods=['POST'])
def add_to_excel():
    data = request.get_json()

    # Automatically generate the timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    ph_value = data.get('ph_value')
    turbidity = data.get('turbidity')
    temperature = data.get('temperature')
    flow_value = data.get('flow_value')

    if not all([ph_value, turbidity, temperature, flow_value]):
        return jsonify({"error": "Missing required parameters"}), 400

    # File name for the Excel file
    file_name = "data.xlsx"
    workbook = load_or_create_excel(file_name)
    sheet = workbook.active

    # Add the data to the Excel sheet
    sheet.append([timestamp, ph_value, turbidity, temperature, flow_value])
    workbook.save(file_name)

    return jsonify({"message": "Data added to Excel successfully!"}), 200

# API route to retrieve the latest row from the Excel file
@app.route('/get_entry', methods=['GET'])
def get_latest_entry():
    # File name for the Excel file
    file_name = "data.xlsx"
    
    if not os.path.exists(file_name):
        return jsonify({"error": "Excel file not found"}), 404

    workbook = openpyxl.load_workbook(file_name)
    sheet = workbook.active

    if sheet.max_row < 2:  # Check if there are any rows other than the header
        return jsonify({"error": "No data available"}), 404

    # Get the latest row (excluding the header)
    latest_row = list(sheet.iter_rows(values_only=True))[-1]
    
    # Create an object for the latest entry
    latest_entry = {
        "timestamp": latest_row[0],
        "ph_value": latest_row[1],
        "turbidity": latest_row[2],
        "temperature": latest_row[3],
        "flow_value": latest_row[4]
    }

    return jsonify({"latest_entry": latest_entry}), 200

@app.route('/predict_latest', methods=['GET'])
def predict_latest():
    file_name = "data.xlsx"
    if not os.path.exists(file_name):
        return jsonify({"error": "Excel file not found"}), 404

    workbook = openpyxl.load_workbook(file_name)
    sheet = workbook.active
    if sheet.max_row < 2:
        return jsonify({"error": "No data available"}), 404

    latest_row = list(sheet.iter_rows(values_only=True))[-1]

    # Build the features dict in the exact shape expected by ML.predict().
    # Note: column indices come from how we append rows in /add_to_excel:
    # [Timestamp, Ph_Value, Turbidity, Temperature, Flow_Value]
    features = {
        # ML expects the following keys: temperature_C, pH, turbidity_NTU, flow_m_s, trash_detected
        'temperature_C': latest_row[3],
        'pH': latest_row[1],
        'turbidity_NTU': latest_row[2],
        'flow_m_s': latest_row[4],
        # The spreadsheet currently doesn't include trash_detected, so we default to 0 (No).
        'trash_detected': 0
    }

    # Call the ML.predict function imported from ML.py. This will use the
    # pre-loaded artifacts (model, scaler, encoders) to produce decoded labels
    # and numeric values for each output column.
    try:
        prediction = predict(features)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Return the prediction as JSON to the client
    return jsonify({"prediction": prediction}), 200

@app.route('/get_database', methods=['GET'])
def get_database():
    file_name = "data.xlsx"
    
    if not os.path.exists(file_name):
        return jsonify({"error": "Excel file not found"}), 404
    
    try:
        return send_file(file_name, as_attachment=True)
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    # Start the Flask app on a specific port
    port = 5001

    # Set your Ngrok auth token from environment
    ngrok_key = os.getenv("NGROK_AUTH_TOKEN")
    ngrok.set_auth_token(ngrok_key)

    # Open an Ngrok tunnel
    public_url = ngrok.connect(port).public_url
    print(f"Ngrok tunnel URL: {public_url}")
    print("ESP32 can now communicate with this server through ngrok")

    # Start the Flask server with external access
    app.run(host='0.0.0.0', port=port)
