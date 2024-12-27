import base64
import cv2
import math
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from flask import Flask, request, jsonify
import traceback
from flask_cors import CORS  # Import CORS
from src.LipShapeAnalyzer import LipShapeAnalyzer
from src.NoseShapeAnalyzer import NoseAnalyzer
from src.EyesShapeAnalyzer import EyeShapeAnalyzer
from src.BodyShapeAnalyzer import BodyShapeAnalyzer
from src.FaceShapeAnalyzer import FaceAnalyzer

from aliyun_upload import upload_to_oss

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
# You can also restrict it to a specific origin:
# CORS(app, origins=["http://localhost:3000"])

# Load the YOLO model (use a pre-trained model, e.g., yolov8n)
model = YOLO('yolov8n-face.pt')  # You can use other YOLOv8 models, like 'yolov8m.pt', etc.

# @app.route('/body_input', method=['POST'])
# def body_input(arg1, arg2, arg3):
#     return

# @app.route('/body_analyze', method=['POST'])
# def body_analyze(image):
#     return

@app.route('/pdf-upload', methods=['GET'])
def pdf_upload():
    result = upload_to_oss()
    return jsonify(result)

@app.route('/mediapipe-detect', methods=['POST'])
def mediapipe_detect():
    try:
        result = {}
        data = request.get_json()

        # Decode the base64 image data
        image_data = data.get("image")
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Remove the base64 header (e.g., "data:image/png;base64,")
        image_data = image_data.split(",")[1]

        # Convert the base64 string to a numpy array
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Use MediaPipe to process the image
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if results.multi_face_landmarks:
                face_landmarks = []
                for face_landmark in results.multi_face_landmarks:
                    landmarks = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face_landmark.landmark]
                    face_landmarks.append(landmarks)
                result["landmarks"] = face_landmarks
            else:
                result["landmarks"] = []

            return jsonify(result)
    except Exception as err:
        tb_str = traceback.format_exc()
        return jsonify({"error": f"Exception while analyzing: {str(err)} - {tb_str}"}), 400
    
@app.route('/body-analyze', methods=['POST'])
def body_analyze():
    try:
        result = {}
        data = request.get_json()

        # Decode the base64 image data
        image_data = data.get("image")
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Remove the base64 header (e.g., "data:image/png;base64,")
        image_data = image_data.split(",")[1]

        # Convert the base64 string to a numpy array
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Analyze body shape
        body_analyzer = BodyShapeAnalyzer(image)
        body_analyzer.image = image  # 使用已经解码的图像
        body_analyzer.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        body_analyzer.detect_landmarks()
        body_analyzer.calculate_head_shoulder_ratio()
        body_analyzer.calculate_body_proportion()

        result = {
            "body_shape_info": body_analyzer.result
        }

        return jsonify(result), 200

        # return jsonify(result)
    except Exception as err:
        tb_str = traceback.format_exc()
        return jsonify({"error": f"Exception while analyzing: {str(err)} - {tb_str}"}), 400
    
@app.route('/face-analyze', methods=['POST'])
def face_analyze():
    try:
        result = {}
        data = request.get_json()

        # Decode the base64 image data
        image_data = data.get("image")
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Remove the base64 header (e.g., "data:image/png;base64,")
        image_data = image_data.split(",")[1]

        # Convert the base64 string to a numpy array
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Analyze the lip shape
        lip_analyzer = LipShapeAnalyzer(image)
        lip_analyzer.detect_landmarks()
        lip_analyzer.analyze_lip_shape()

        # Analyze the nose shape
        nose_analyzer = NoseAnalyzer(image)
        nose_analyzer.detect_landmarks()
        # 1. Nose width analysis
        _, nose_width = nose_analyzer.calculate_distance_ratio()
        # 2. Nose bridge position analysis
        nose_bridge_result = nose_analyzer.analyze_nose_bridge(threshold=20)
        # 3. Nose wing curvature analysis
        nose_wing_curvature = nose_analyzer.analyze_nose_wing_curvature()

        # Analyze eye shape
        eye_analyzer = EyeShapeAnalyzer(image)
        eye_analyzer.detect_landmarks()
        eye_analyzer.analyze_eye_shape()

        # FaceShapeAnalyze 三庭五眼和气质
        face_analyzer = FaceAnalyzer(image)
        face_analyzer.analyze()

        # # Analyze body shape
        # body_analyzer = BodyShapeAnalyzer(image=None)
        # body_analyzer.image = image  # 使用已经解码的图像
        # body_analyzer.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # body_analyzer.calculate_head_shoulder_ratio()
        # body_analyzer.calculate_body_proportion()
      
        # Collect the analysis results
        result = {
            "lip_shape": lip_analyzer.result,
            "Lips_detailed_info": {
                "唇形": lip_analyzer.judge_round_wide_lip()[1],
                "厚薄":lip_analyzer.judge_thin_thick_lip()[1],
                "唇形":lip_analyzer.judge_m_lip()[1],
                "右嘴角": lip_analyzer.right_angle,
                "左嘴角": lip_analyzer.left_angle,
            },
            "nose_detailed_info": {
                "鼻翼宽窄判断": nose_width,
                "山根位置": nose_bridge_result,
                "鼻翼曲直判断": nose_wing_curvature,
            },
            "eye_detailed_info": eye_analyzer.result, 
            "Face_shape_info": face_analyzer.result,
            #  "body_shape_info": body_analyzer.result
        }

        return jsonify(result), 200

        # return jsonify(result)
    except Exception as err:
        tb_str = traceback.format_exc()
        return jsonify({"error": f"Exception while analyzing: {str(err)} - {tb_str}"}), 400
    
@app.route('/yolo-detect', methods=['POST'])
def yolo_detect():
    try:
        data = request.get_json()

        # Decode the base64 image data
        image_data = data.get("image")
        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Remove the base64 header (e.g., "data:image/png;base64,")
        image_data = image_data.split(",")[1]

        # Convert the base64 string to a numpy array
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLO detection
        results = model(image)

        # Initialize an empty list to store bounding box data
        bounding_boxes = []

        # Class ID for "person" (if you're using a general YOLO model)
        # For example, in COCO dataset, "person" is usually class 0.
        FACE_CLASS_ID = 0  # Replace with the correct ID for "face" if available in your model

        # Iterate over detected boxes
        for box in results[0].boxes:
            # Check if the detected class is for "face" or "person"
            detected_class = int(box.cls.cpu().numpy()[0])
            if detected_class == FACE_CLASS_ID:
                # Extract the box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf.cpu().numpy()[0]

                # Add each box to the list if it is a face
                bounding_boxes.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "confidence": float(confidence),
                    "class": detected_class
                })

        # Return detected bounding boxes as JSON
        return jsonify({"bounding_boxes": bounding_boxes})

    except Exception as err:
        tb_str = traceback.format_exc()
        return jsonify({"error": f"Exception while analyzing: {str(err)} - {tb_str}"}), 400



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
