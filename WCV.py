import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# 1. Load Images
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f'Error: Could not load image {image_path}')
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

dataset_dir = r'D:\Computer Vision Workshop\dataset\images'
images = []
labels = []

for root, dirs, files in os.walk(dataset_dir):
    if len(files) == 0:
        continue
    for f in files:
        _, image = load_image(os.path.join(root, f))
        if image is None:
            continue
        images.append(image)
        labels.append(os.path.basename(root))

print(f'Total images loaded: {len(labels)}')

# 2. Face Detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces

def crop_faces(image_gray, faces, return_all=False):
    cropped_faces = []
    selected_faces = []
    if len(faces) > 0:
        if return_all:
            for x, y, w, h in faces:
                selected_faces.append((x, y, w, h))
                cropped_faces.append(image_gray[y:y+h, x:x+w])
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            selected_faces.append((x, y, w, h))
            cropped_faces.append(image_gray[y:y+h, x:x+w])
    return cropped_faces, selected_faces

# 3. Preprocessing
face_size = (128, 128)

def resize_and_flatten(face):
    face_resized = cv2.resize(face, face_size)
    face_flattened = face_resized.flatten()
    return face_flattened

x = []
y = []

for image, label in zip(images, labels):
    faces = detect_faces(image)
    cropped_faces, _ = crop_faces(image, faces)
    if len(cropped_faces) > 0:
        face_flattened = resize_and_flatten(cropped_faces[0])
        x.append(face_flattened)
        y.append(label)

x = np.array(x)
y = np.array(y)

print(f'Dataset shape: {x.shape}')

# 4. Train Model
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=177, stratify=y)

class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self
    def transform(self, X):
        return X - self.mean_face

pipe = Pipeline([
    ('centering', MeanCentering()),
    ('pca', PCA(svd_solver='randomized', whiten=True, random_state=177)),
    ('svc', SVC(kernel='linear', random_state=177))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Save Model
with open('eigenface_pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# 6. Real-Time Face Recognition
def get_eigenface_score(X):
    X_pca = pipe[:2].transform(X)
    eigenface_scores = np.max(pipe[2].decision_function(X_pca), axis=1)
    return eigenface_scores

def eigenface_prediction(image_gray):
    faces = detect_faces(image_gray)
    cropped_faces, selected_faces = crop_faces(image_gray, faces)
    
    if len(cropped_faces) == 0:
        return None, None, None
    
    X_face = []
    for face in cropped_faces:
        face_flattened = resize_and_flatten(face)
        X_face.append(face_flattened)
    
    X_face = np.array(X_face)
    labels = pipe.predict(X_face)
    scores = get_eigenface_score(X_face)
    
    return scores, labels, selected_faces

def draw_text(image, label, score, pos=(0, 0), 
              font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, font_thickness=2,
              text_color=(0, 0, 0), text_color_bg=(0, 255, 0)):
    x, y = pos
    score_text = f'Score: {score:.2f}'
    (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(image, (x, y-h1-h2-25), (x + max(w1, w2)+20, y), text_color_bg, -1)
    cv2.putText(image, label, (x+10, y-10), font, font_scale, text_color, font_thickness)
    cv2.putText(image, score_text, (x+10, y-h2-15), font, font_scale, text_color, font_thickness)

def draw_result(image, scores, labels, coords):
    result_image = image.copy()
    for (x, y, w, h), label, score in zip(coords, labels, scores):
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_text(result_image, label, score, pos=(x, y))
    return result_image

# 7. Start Webcam
cap = cv2.VideoCapture(0)  # 0 = default camera

if not cap.isOpened():
    print('Error: Cannot open camera')
    exit()

print("Press ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print('Error: Cannot read frame')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores, labels, coords = eigenface_prediction(gray)

    if labels is not None:
        frame = draw_result(frame, scores, labels, coords)

    cv2.imshow('Real-Time Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to stop
        break

cap.release()
cv2.destroyAllWindows()
