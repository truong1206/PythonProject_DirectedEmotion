# app.py
from flask import Flask, render_template, Response, request, redirect, url_for, send_from_directory, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('data/emotion_detection_model.h5')
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cap = cv2.VideoCapture(0)
show_window = True

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Add this block to create the 'uploads' directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames(source_type='camera', filename=None):
    while True:
        if source_type == 'camera':
            success, frame = cap.read()
        elif source_type == 'image' and filename:
            frame = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            break

        if not success:
            break
        else:
            if show_window and source_type == 'camera':
                face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    # Draw rectangle around the face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                    roi_gray = gray[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                    prediction = model.predict(cropped_img)
                    max_index = int(np.argmax(prediction))
                    emotion_text = emotion_dict[max_index]
                    confidence = np.max(prediction) * 100  # Confidence in percentage
                    cv2.putText(frame, f"Emotion: {emotion_text}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/browser')
def browser_page():
    return render_template('browser.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/browser', methods=['GET', 'POST'])
def browser():
    if request.method == 'POST':
        file = request.files['myfile']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print(f"Received file: {filename}")
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Perform emotion detection on the uploaded image
            detected_emotion, modified_image_path = detect_emotion(file_path)

            return jsonify({'emotion': detected_emotion, 'modified_image_path': modified_image_path})

    return render_template('browser.html')


def detect_emotion(image_path):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Set a default value for emotion_text
        emotion_text = "Unknown"

        for (x, y, w, h) in faces:
            # Draw rectangle around the face
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Extract the face region
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

            # Perform emotion prediction
            prediction = model.predict(cropped_img)
            max_index = int(np.argmax(prediction))
            emotion_text = emotion_dict[max_index]
            confidence = np.max(prediction) * 100  # Confidence in percentage

            # Add text to the image
            cv2.putText(img, f"Emotion: {emotion_text}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(img, f"Confidence: {confidence:.2f}%", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)

        # Save the image with the bounding box
        modified_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'modified_image.jpg')
        cv2.imwrite(modified_image_path, img)

        # Return the detected emotion and modified image path
        return emotion_text, modified_image_path

    except Exception as e:
        print("Error detecting emotion:", str(e))
        return "Error", None
if __name__ == '__main__':
    app.run(debug=True)
