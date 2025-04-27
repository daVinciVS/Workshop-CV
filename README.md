# Face Recognition using Eigenfaces

Simple face detection and recognition system using **OpenCV**, **Scikit-learn**, and **Matplotlib**, based on the **Eigenfaces** method.

---

## Project Structure
project-folder/ ├── dataset/ │ └── images/ │ ├── George_W_Bush/ │ ├── Laura_Bush/ │ ├── Robert_Downey_Jr/ │ ├── Serena_Williams/ │ ├── Tom_Cruise/ │ ├── Vaughn_D_Kenneth/ │ └── Vladimir_Putin/ │ ├── eigenface_pipeline.pkl ├── main.py └── README.md

---

###  Installation

1. Install Python 3.13.
2. Install required libraries:

python -m pip install opencv-python numpy matplotlib scikit-learn

## How to Run

### 1. Train the Model



Edit `main.py` to set the path to your dataset:

dataset_dir = 'D:/Computer Vision Workshop/dataset/images'


Run the training script:


python main.py


This will:
Detect faces from the dataset
Train an Eigenface-based SVM classifier
Save the trained model pipeline to eigenface_pipeline.pkl


# 2. Real-Time Recognition with Webcam


After training:
Load the trained model
Capture frames from your webcam
Detect and recognize faces in real-time


Example webcam prediction code:


import cv2
import pickle

# Load trained pipeline
with open('eigenface_pipeline.pkl', 'rb') as f:
    pipe = pickle.load(f)

def eigenface_prediction(image_gray): 
    faces = detect_faces(image_gray) 
    cropped_faces, selected_faces = crop_faces(image_gray, faces) 
    if len(cropped_faces) == 0: 
        return 'No face detected.' 
    X_face = [resize_and_flatten(face) for face in cropped_faces]
    X_face = np.array(X_face)
    labels = pipe.predict(X_face) 
    scores = get_eigenface_score(X_face)
    return scores, labels, selected_faces

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = eigenface_prediction(gray)
    if result != 'No face detected.':
        scores, labels, coords = result
        frame = draw_result(frame, scores, labels, coords)
    cv2.imshow('Real-Time Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



Press Q on your keyboard to quit the webcam window.


## Notes
Adding New People: Add new folders inside dataset/images/ and retrain.

No Face Detected?: Ensure good lighting and correct webcam settings.

Model Size: PCA is used to reduce image dimensionality for faster processing.
