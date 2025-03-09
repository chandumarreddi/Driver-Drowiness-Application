from flask import Flask, render_template, Response, request, redirect, url_for, session
import cv2
import numpy as np
import mediapipe as mp
import time
from scipy.spatial import distance
from tensorflow.keras.models import load_model
from pygame import mixer
import json
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Initialize the sound mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

# Load CNN model for eye classification (if used)
model = load_model('CNN__model.h5')

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Define landmark indices for eyes and mouth (based on MediaPipe)
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
MOUTH_INDICES = [78, 95, 88, 191, 80, 81, 13, 311, 308, 402, 318, 324]

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye_points):
    A = distance.euclidean(eye_points[1], eye_points[5])
    B = distance.euclidean(eye_points[2], eye_points[4])
    C = distance.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Mouth Aspect Ratio (MAR)
def mouth_aspect_ratio(mouth_points):
    A = distance.euclidean(mouth_points[1], mouth_points[5])
    B = distance.euclidean(mouth_points[2], mouth_points[4])
    C = distance.euclidean(mouth_points[0], mouth_points[3])
    mar = (A + B) / (2.0 * C)
    return mar

# Define EAR and MAR thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.25
ALARM_THRESHOLD = 1  # Alarm triggers after 10 seconds of continuous closure

# Initialize variables
score = 0
alarm_playing = False
closed_eye_start_time = None
video_feed_active = False  # Flag to control video feed
cap = None  # Video capture object

# Path to save user data
USER_DATA_FILE = 'users1.json'

# Load existing user data or create an empty list
if os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, 'r') as file:
        users = json.load(file)
else:
    users = []

def generate_frames():
    global score, alarm_playing, closed_eye_start_time, video_feed_active, cap
    cap = cv2.VideoCapture(0)  # Initialize camera here
    while video_feed_active:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Get eye and mouth landmarks
                left_eye_pts = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE_INDICES])
                right_eye_pts = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE_INDICES])
                mouth_pts = np.array([(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in MOUTH_INDICES])

                # Calculate EAR and MAR
                left_EAR = eye_aspect_ratio(left_eye_pts)
                right_EAR = eye_aspect_ratio(right_eye_pts)
                mouth_MAR = mouth_aspect_ratio(mouth_pts)

                avg_EAR = (left_EAR + right_EAR) / 2.0

                # Draw landmarks
                for (x, y) in np.concatenate((left_eye_pts, right_eye_pts, mouth_pts), axis=0):
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Detect closed eyes using EAR
                if avg_EAR < EAR_THRESHOLD:
                    if closed_eye_start_time is None:
                        closed_eye_start_time = time.time()
                    elif time.time() - closed_eye_start_time >= 1:  # Count time properly
                        score = int(time.time() - closed_eye_start_time)
                else:
                    closed_eye_start_time = None
                    score = 0  # Reset score when eyes open

                # Display EAR, MAR, and Score
                cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"MAR: {mouth_MAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, f"Score: {score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Check if alarm should play
                if score >= ALARM_THRESHOLD:
                    if not alarm_playing:
                        sound.play(-1)
                        alarm_playing = True
                else:
                    if alarm_playing:
                        sound.stop()
                        alarm_playing = False

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Release the camera when the feed stops
    cap.release()

@app.route('/')
def index():
    if 'username' in session:
        # Find the logged-in user's data
        user_data = next((user for user in users if user['email'] == session.get('email')), None)
        if user_data:
            return render_template('index.html', username=session['username'], user_data=user_data)
    return redirect(url_for('login'))
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        user_data = {
            'firstName': request.form['firstName'],
            'lastName': request.form['lastName'],
            'email': request.form['email'],
            'password': request.form['password'],
            'phone': request.form['phone'],
            'drivingLicense': request.form['drivingLicense'],
            'vehicleType': request.form['vehicleType'],
            'driverNumber': request.form['driverNumber'],  # New field
            'driverType': request.form['driverType'],
            'emergencyContact': request.form['emergencyContact']
        }

        # Save user data to JSON file
        users.append(user_data)
        with open(USER_DATA_FILE, 'w') as file:
            json.dump(users, file, indent=4)

        # Redirect to login page after registration
        return redirect(url_for('login'))

    return render_template('register.html')
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Check if user exists
        for user in users:
            if user['email'] == email and user['password'] == password:
                session['username'] = user['firstName']
                session['email'] = user['email']  # Store email in session
                return redirect(url_for('index'))

        return "Invalid email or password. Please try again."

    return render_template('login.html')
@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove username from session
    return redirect(url_for('login'))

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_feed', methods=['POST'])
def start_feed():
    global video_feed_active
    video_feed_active = True
    return {'status': 'started'}

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    global video_feed_active, cap
    video_feed_active = False
    if cap is not None:
        cap.release()  # Release the camera
    return {'status': 'stopped'}

@app.route('/check_alarm')
def check_alarm():
    global alarm_playing
    return {'alarm': alarm_playing}

if __name__ == '__main__':
    app.run(debug=True)