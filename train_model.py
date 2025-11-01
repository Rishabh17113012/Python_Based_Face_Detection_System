import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import face_recognition

# ==============================
# Function to extract face embeddings
# ==============================
def extract_face_embedding(image):
    face_encodings = face_recognition.face_encodings(image)
    return face_encodings[0] if face_encodings else None


# ==============================
# Load dataset and prepare training data
# ==============================
dataset_dir = "dataset"  # Folder containing subfolders (one per person)

X = []
y = []

if not os.path.exists(dataset_dir):
    print(f"[ERROR] Dataset folder '{dataset_dir}' not found.")
    exit(1)

people = [p for p in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, p))]
if not people:
    print("[ERROR] No person folders found inside the dataset directory.")
    exit(1)

for person_name in people:
    person_dir = os.path.join(dataset_dir, person_name)
    image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    processed_count = 0
    for image_name in image_files:
        img_path = os.path.join(person_dir, image_name)
        image = face_recognition.load_image_file(img_path)

        # Extract face embedding
        face_embedding = extract_face_embedding(image)
        if face_embedding is not None:
            X.append(face_embedding)
            y.append(person_name)
            processed_count += 1

    print(f"[INFO] Processed {processed_count} images for {person_name}")

# ==============================
# Validate dataset before training
# ==============================
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("[ERROR] No valid face encodings found. Make sure images contain clear, front-facing faces.")
    exit(1)

unique_labels = np.unique(y)
if len(unique_labels) < 2:
    print(f"[ERROR] Need at least two different people to train. Found only: {list(unique_labels)}")
    exit(1)

# ==============================
# Train SVM model
# ==============================
print(f"[INFO] Training model with {len(X)} samples from {len(unique_labels)} people...")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

classifier = SVC(kernel='linear', probability=True)
classifier.fit(X, y_encoded)

# ==============================
# Save trained model
# ==============================
joblib.dump(classifier, 'face_recognition_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

print("[SUCCESS] Model training complete and saved successfully.")
