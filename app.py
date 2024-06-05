import tensorflow as tf
from flask import Flask, request
import numpy as np
import mediapipe as mp
import cv2
import base64
from flask import jsonify, render_template
from flask_socketio import SocketIO, send
from datetime import datetime

model = tf.keras.models.load_model('my_model2.keras')

actions = np.array(['hola', 'jugar', 'a', 'e', 'gracias', 'ir','yo', 'neutral','tu','donde', 'dias', 'adios', 'nombre', 'de nada'])
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
    right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
    left = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
    return np.concatenate([pose, face, right, left])

def decode_image(img):
    image_data = img.split(",")[1]
    image_bytes = base64.b64decode(image_data)
    image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
    return frame

holistic =  mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

app = Flask(__name__)
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")


frames = []
sequence = []
predictions = []
threshold = 0.5

@socketio.on("connect")
def on_connect():
    print("Cliente conectado")

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on("message")
def on_message(data):   
   
    global sequence
    #print(data["frames"])
    image_base64 = data["frames"]
    #start_time = datetime.now()
    frame = decode_image(image_base64)
    #end_time = datetime.now()
    #print(end_time - start_time)
    
    image, results = mediapipe_detection(frame, holistic)
    keypoints = extract_keypoints(results)
    print(keypoints)
    sequence.append(keypoints)
    
    if len(sequence) == 30:
        #print(sequence[1])
        res = model.predict(np.expand_dims(sequence, axis=0))[0]
        prediction = actions[np.argmax(res)]
        socketio.emit("prediction", {"prediction": prediction})
        sequence = []

   
if __name__ == "__main__":
    socketio.run(app, host="localhost", port=5000, allow_unsafe_werkzeug=True)