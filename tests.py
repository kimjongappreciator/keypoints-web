import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import cv2
import base64
from app import mediapipe_detection, decode_image, extract_keypoints, app

class TestAppFunctions(unittest.TestCase):

    def test_decode_image_success(self):
        # Simular una imagen en base64
        img = base64.b64encode(cv2.imencode('.jpg', np.zeros((100, 100, 3), dtype=np.uint8))[1]).decode('utf-8')
        frame = decode_image(img)
        self.assertIsNotNone(frame)
        self.assertEqual(frame.shape, (100, 100, 3))

    def test_decode_image_failure(self):
        # Probar con una entrada inválida
        img = "invalid_base64"
        frame = decode_image(img)
        self.assertIsNone(frame)

    
    def test_extract_keypoints_no_results(self):
        # Probar cuando no hay landmarks detectados
        results = MagicMock()
        results.pose_landmarks = None
        results.right_hand_landmarks = None
        results.left_hand_landmarks = None
        results.face_landmarks = None

        keypoints = extract_keypoints(results)
        self.assertEqual(len(keypoints), 1662)
        self.assertTrue(np.all(keypoints == 0))

    def test_extract_keypoints_with_landmarks(self):
        # Probar cuando hay landmarks válidos
        results = MagicMock()
        results.pose_landmarks.landmark = [MagicMock(x=1, y=2, z=3, visibility=0.9)] * 33
        results.right_hand_landmarks.landmark = [MagicMock(x=1, y=2, z=3)] * 21
        results.left_hand_landmarks.landmark = [MagicMock(x=1, y=2, z=3)] * 21
        results.face_landmarks.landmark = [MagicMock(x=1, y=2, z=3)] * 468

        keypoints = extract_keypoints(results)
        self.assertEqual(len(keypoints), 1662)
    

    @patch("app.model.predict")
    def test_model_prediction(self, mock_predict):
        mock_predict.return_value = np.array([[0.1, 0.9]])  # Simular una predicción
        sequence = [np.zeros(1662)] * 30  # Simular una secuencia completa de keypoints
        res = mock_predict(np.expand_dims(sequence, axis=0))[0]
        prediction = np.argmax(res)
        self.assertEqual(prediction, 1)  # Simular que predice la segunda acción

if __name__ == '__main__':
    unittest.main()
