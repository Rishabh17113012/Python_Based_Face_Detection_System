from flask import Flask, render_template, request, redirect, url_for, jsonify, Response
import os
import cv2
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import face_recognition
import time

app = Flask(__name__)

# Global variables
classifier = None
label_encoder = None
camera = None
is_collecting = False
is_recognizing = False
current_person_name = ""
collection_count = 0

# Load models if they exist
try:
    classifier = joblib.load('face_recognition_model.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
    print("[INFO] Models loaded successfully")
except Exception as e:
    print(f"[INFO] No models found: {str(e)}")

def get_camera():
    """Get or create camera instance"""
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def release_camera():
    """Release camera resource"""
    global camera
    if camera is not None and camera.isOpened():
        camera.release()
        camera = None

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Data Collection - Start
@app.route('/data_collection', methods=['POST'])
def data_collection():
    global is_collecting, current_person_name, collection_count
    
    person_name = request.form.get('person_name')
    if not person_name:
        return jsonify({"status": "error", "message": "Please provide a name"}), 400
    
    current_person_name = person_name
    is_collecting = True
    collection_count = 0
    
    # Create directory
    save_path = f"dataset/{person_name}"
    os.makedirs(save_path, exist_ok=True)
    
    return jsonify({
        "status": "success",
        "message": f"Data collection started for {person_name}"
    })

# Stop Collection
@app.route('/stop_collection', methods=['POST'])
def stop_collection():
    global is_collecting, collection_count
    is_collecting = False
    count = collection_count
    release_camera()
    return jsonify({
        "status": "success",
        "message": f"Collection stopped. {count} images saved."
    })

# Video feed for data collection
@app.route('/video_feed_collection')
def video_feed_collection():
    def generate():
        global collection_count, is_collecting, current_person_name
        
        cam = get_camera()
        frame_count = 0
        save_path = f"dataset/{current_person_name}"
        
        while is_collecting:
            success, frame = cam.read()
            if not success:
                break
            
            # Save every 15th frame (about 2 images per second at 30fps)
            if frame_count % 15 == 0 and collection_count < 50:
                img_name = f"{save_path}/{current_person_name}_{collection_count}.jpg"
                cv2.imwrite(img_name, frame)
                collection_count += 1
                print(f"[INFO] Saved: {img_name}")
            
            # Add overlay text
            text = f"Collecting: {current_person_name} | Saved: {collection_count}/50"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            
            # Draw a face guide rectangle
            height, width = frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            rect_size = 200
            cv2.rectangle(frame, 
                        (center_x - rect_size, center_y - rect_size),
                        (center_x + rect_size, center_y + rect_size),
                        (0, 255, 255), 2)
            cv2.putText(frame, "Position face here", (center_x - 100, center_y - rect_size - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            frame_count += 1
            
            # Stop after collecting enough images
            if collection_count >= 50:
                is_collecting = False
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        release_camera()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Model Training Route
@app.route('/train_model', methods=['POST'])
def train_model():
    global classifier, label_encoder

    print("[INFO] Starting model training...")
    
    X = []
    y = []
    dataset_dir = "dataset"

    if not os.path.exists(dataset_dir):
        return jsonify({"status": "error", "message": "No dataset found"}), 400

    def extract_face_embedding(image):
        face_encodings = face_recognition.face_encodings(image)
        return face_encodings[0] if face_encodings else None

    person_count = 0
    total_images = 0
    
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)

        if os.path.isdir(person_dir):
            person_count += 1
            image_count = 0
            
            for image_name in os.listdir(person_dir):
                if image_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(person_dir, image_name)
                    try:
                        image = face_recognition.load_image_file(img_path)
                        face_embedding = extract_face_embedding(image)

                        if face_embedding is not None:
                            X.append(face_embedding)
                            y.append(person_name)
                            image_count += 1
                    except Exception as e:
                        print(f"[WARNING] Could not process {img_path}: {e}")
            
            total_images += image_count
            print(f"[INFO] Processed {image_count} images for {person_name}")

    if len(X) == 0:
        return jsonify({"status": "error", "message": "No valid faces found in dataset"}), 400

    X = np.array(X)
    y = np.array(y)

    print(f"[INFO] Training with {len(X)} samples from {person_count} people")

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Train SVM with better parameters for accuracy
    classifier = SVC(kernel='linear', probability=True, C=1.0)
    classifier.fit(X, y_encoded)

    # Save models
    joblib.dump(classifier, 'face_recognition_model.pkl')
    joblib.dump(label_encoder, 'label_encoder.pkl')

    print("[INFO] Model training complete")
    
    return jsonify({
        "status": "success",
        "message": f"Model trained successfully with {total_images} images from {person_count} people"
    })

# Start Recognition
@app.route('/recognize', methods=['POST'])
def recognize():
    global is_recognizing
    
    if classifier is None or label_encoder is None:
        return jsonify({"status": "error", "message": "Train model first"}), 400
    
    is_recognizing = True
    return jsonify({
        "status": "success",
        "message": "Recognition started"
    })

# Stop Recognition
@app.route('/stop_recognition', methods=['POST'])
def stop_recognition():
    global is_recognizing
    is_recognizing = False
    release_camera()
    return jsonify({
        "status": "success",
        "message": "Recognition stopped"
    })

# Video feed for recognition
@app.route('/video_feed_recognition')
def video_feed_recognition():
    def generate():
        global is_recognizing
        
        cam = get_camera()
        
        while is_recognizing:
            success, frame = cam.read()
            if not success:
                break
            
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces with better accuracy settings
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Recognize each face
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                try:
                    # Get prediction with probability
                    y_pred = classifier.predict([face_encoding])
                    proba = classifier.predict_proba([face_encoding])
                    confidence = np.max(proba) * 100
                    
                    name = label_encoder.inverse_transform(y_pred)[0]
                    
                    # Only show name if confidence is high enough
                    if confidence > 60:  # Threshold for better accuracy
                        display_text = f"{name} ({confidence:.1f}%)"
                        color = (0, 255, 0)  # Green for recognized
                    else:
                        display_text = f"Unknown ({confidence:.1f}%)"
                        color = (0, 0, 255)  # Red for unknown
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    
                    # Draw label background
                    cv2.rectangle(frame, (left, top - 35), (right, top), color, cv2.FILLED)
                    
                    # Put text
                    cv2.putText(frame, display_text, (left + 6, top - 6),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                except Exception as e:
                    print(f"[ERROR] Recognition error: {e}")
            
            # Add instructions
            cv2.putText(frame, "Recognition Active", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        release_camera()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Get dataset info
@app.route('/dataset_info')
def dataset_info():
    dataset_dir = "dataset"
    if not os.path.exists(dataset_dir):
        return jsonify({"people": [], "total": 0})
    
    people = []
    for person_name in os.listdir(dataset_dir):
        person_dir = os.path.join(dataset_dir, person_name)
        if os.path.isdir(person_dir):
            image_count = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            people.append({"name": person_name, "images": image_count})
    
    return jsonify({"people": people, "total": len(people)})

if __name__ == "__main__":
    app.run(debug=True, threaded=True)