import tensorflow as tf
from flask import Flask, jsonify, render_template, request
import numpy as np
import mediapipe as mp
import cv2
import base64
from flask_socketio import SocketIO
from datetime import datetime
import database as dbase
import os

# Manejo de errores en la carga del modelo
try:
    model = tf.keras.models.load_model('my_model3.keras')
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model = None

actions = np.array(['hola', 'jugar', 'a', 'e', 'gracias', 'ir','yo', 'neutral','tu','donde', 'dias', 'adios', 'nombre', 'de nada', 'ordenar', 'guardar'])
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    except Exception as e:
        print(f"Error en mediapipe_detection: {e}")
        return image, None

def draw_landmarks(image, results):
    try:
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    except Exception as e:
        print(f"Error en draw_landmarks: {e}")

def extract_keypoints(results):
    try:
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
        right = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(63)
        left = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(63)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
        return np.concatenate([pose, face, right, left])
    except Exception as e:
        print(f"Error en extract_keypoints: {e}")
        return np.zeros(1662)  # Tamaño total de los keypoints en caso de error

def decode_image(img):
    try:
        image_bytes = base64.b64decode(img)
        image_buffer = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error al decodificar la imagen: {e}")
        return None

holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

app = Flask(__name__)
socketio = SocketIO(app)
socketio.init_app(app, cors_allowed_origins="*")

# Manejo de errores en la conexión a la base de datos
try:
    db = dbase.dbConnection()
    collection = db["translation_log"]
except Exception as e:
    print(f"Error en la conexión a la base de datos: {e}")
    collection = None

frames = []
sequence = []
predictions = []
threshold = 0.5

@socketio.on("connect")
def on_connect():
    print("Cliente conectado")

@socketio.on("disconnect")
def on_disconnect():
    print("Cliente desconectado")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/test", methods=["POST"])
def test():
    global sequence
    try:
        data = request.get_json()
        imgbase64 = data["frames"]
        frame = decode_image(imgbase64)

        if frame is None:
            return jsonify({"error": "Error al decodificar la imagen"}), 400
        
        image, results = mediapipe_detection(frame, holistic)
        if results is None:
            return jsonify({"error": "Error en la detección de Mediapipe"}), 400

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            prediction = actions[np.argmax(res)]
            print(prediction)
            socketio.emit("prediction", {"prediction": prediction})
            sequence = []
        return jsonify({"message": "Imagen recibida"})
    except Exception as e:
        print(f"Error en el endpoint /test: {e}")
        return jsonify({"error": "Ocurrió un error"}), 500

@app.route("/write", methods=["POST"])
def add_translation():
    try:
        datosjson = request.get_json()
        if collection is not None:
            collection.insert_one(datosjson)
            return jsonify({"message": "Transacción guardada"})
        else:
            return jsonify({"error": "No se pudo conectar a la base de datos"}), 500
    except Exception as e:
        print(f"Error en el endpoint /write: {e}")
        return jsonify({"error": "Ocurrió un error"}), 500

@app.route("/read", methods=["GET"])
def get_translations():
    try:
        datos = []
        if collection is not None:
            result = collection.find()
            for res in result:
                res['_id'] = str(res['_id'])
                datos.append(res)
            return jsonify({"translations": datos})
        else:
            return jsonify({"error": "No se pudo conectar a la base de datos"}), 500
    except Exception as e:
        print(f"Error en el endpoint /read: {e}")
        return jsonify({"error": "Ocurrió un error"}), 500

@socketio.on("message")
def on_message(data):       
    global sequence
    try:
        image_base64 = data["frames"]
        frame = decode_image(image_base64)

        if frame is None:
            print("Error al decodificar la imagen")
            return
        
        image, results = mediapipe_detection(frame, holistic)
        if results is None:
            print("Error en la detección de Mediapipe")
            return

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            prediction = actions[np.argmax(res)]
            print(prediction)
            socketio.emit("prediction", {"prediction": prediction})
            sequence = []
    except Exception as e:
        print(f"Error en la función on_message: {e}")

if __name__ == "__main__":
    try:
        socketio.run(app, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
    except Exception as e:
        print(f"Error al iniciar la aplicación: {e}")
