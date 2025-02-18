import base64
import cv2
import math
import os
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from flask import Flask, request, jsonify, session
import traceback
from flask_session import Session
from flask_cors import CORS  # Import CORS
from src.LipShapeAnalyzer import LipShapeAnalyzer
from src.NoseShapeAnalyzer import NoseAnalyzer
from src.EyesShapeAnalyzer import EyeShapeAnalyzer
from src.BodyShapeAnalyzer import PoseAnalyzer
from src.BodyShapeAnalyzer import ImageSegmentationProcessor
from src.BodyShapeAnalyzer import PoseSegmentationVisualizer
from src.FaceShapeAnalyzer import FaceAnalyzer
from src.FaceVolumeAnalyzer import FaceVolumeAnalyzer
from datetime import timedelta
from aliyun_upload import upload_to_oss


app = Flask(__name__)
# 设置 Flask `session` 存储
app.config["SECRET_KEY"] = "seasons"  # 设置 SECRET_KEY 以支持加密
app.config["SESSION_TYPE"] = "filesystem"  # 让 session 存在服务器文件系统
app.config["SESSION_PERMANENT"] = True  # 让 session 在多个请求之间持久化
app.config["SESSION_FILE_DIR"] = "./flask_session"  # 指定 session 存储位置
app.config["SESSION_USE_SIGNER"] = True  # 让 session 加密，增强安全性

if not os.path.exists("./flask_session"):
    os.makedirs("./flask_session")

Session(app)  # 初始化 session

print("Flask SESSION_TYPE:", app.config["SESSION_TYPE"])

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 限制16MB
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

@app.route('/pdf-upload', methods=['POST'])
def pdf_upload():
    data = request.json  # Get request JSON
    face_info = data.get('face_info')
    body_info = data.get('body_info')

    result = upload_to_oss(face_info, body_info)
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

        height = data.get("height")
        chest = data.get("chest")
        waist = data.get("waist")
        hips = data.get("hips")

        face_result = data.get("face_result")

        print(face_result)

        # 检查必需的数据是否存在。
        # 【这里三围如果没输入是否可以跳过？身材分析determine_body_shape没有用到三围】 
        # 或者有三围可以直接计算身材？
        missing_data = []
        if height is None:
            missing_data.append("身高")
        if chest is None:
            missing_data.append("胸围")
        if waist is None:
            missing_data.append("腰围")
        if hips is None:
            missing_data.append("臀围")

        if missing_data:
            return jsonify({"error": f"缺少必要的数据: {', '.join(missing_data)}"}), 400
        
        print("-----------------------------------------------")
        print(height, chest, waist, hips)
        print("-----------------------------------------------")

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
        if image is None:
            return jsonify({"error": "Failed to decode image"}), 400
        
        # 处理用户输入的身高
        user_height = height
        if not user_height:
            return jsonify({"error": "No height data provided"}), 400
        user_height = float(user_height)



        # 从 session 读取 ratio_1 和 ratio_5
        eye_ratio_str = face_result['Face_shape_info']['五眼比例']
        eye_ratios = [float(x.strip()) for x in eye_ratio_str.split(":")]  # 去掉空格并转换为 float
        ratio_1 = eye_ratios[0]  # 第一个数
        ratio_5 = eye_ratios[-1]  # 最后一个数
        lip_curve = face_result['Lips_detailed_info']['曲直结果']
        nose_curve = face_result['nose_detailed_info']['鼻型综合曲直']
        eye_curve = face_result['eye_detailed_info']['眼型综合曲直']
        face_curve = face_result['Face_shape_info']['脸型曲直']

        # 检查缺失数据
        missing_data1 = []
        if ratio_1 is None:
            missing_data1.append("ratio_1")
        if ratio_5 is None:
            missing_data1.append("ratio_5")
        if lip_curve is None:
            missing_data1.append("lip_curve")
        if nose_curve is None:
            missing_data1.append("nose_curve")
        if eye_curve is None:
            missing_data1.append("eye_curve")
        if face_curve is None:
            missing_data1.append("face_curve")
        # 如果有缺失的数据，返回具体的缺失项
        if missing_data1:
            return jsonify({"error": f"Missing data in session: {', '.join(missing_data1)}"}), 400
        
        # Analyze body shape
        body_analyzer = PoseAnalyzer(image)
        body_analyzer.analyze()
        body_analyzer.image = image  # 使用已经解码的图像
        body_analyzer.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        
        # 运行量感分析
        face_volume_analyzer = FaceVolumeAnalyzer(image, user_height, lip_curve, nose_curve, eye_curve, face_curve, ratio_1, ratio_5)
        volume_result = face_volume_analyzer.analyze()

        #修改
        try:
            three_d_model = PoseSegmentationVisualizer(image, model_path="selfie_segmenter.tflite")
            three_d_model.process_and_visualize()
        except Exception as e:
            return jsonify({"error": f"3D Model visualization error: {str(e)}"}), 400

        result = {
            "body_shape": body_analyzer.result,
             "body_detailed_info": {
                "头肩比": body_analyzer.result.get('头肩比', '未知'),
                "上下半身比例": body_analyzer.result.get('上下半身比例', '未知'),
                "头肩比判断": body_analyzer.result.get('头肩比判断', '未知'),
                "身材比例判断": body_analyzer.result.get('身材比例判断', '未知'),            
            },
            "three_d_model": three_d_model.result,
            "three_d_model_info": {
                "三围比例": three_d_model.result.get('身材比例(肩：腰：臀)', '未知'),
                "身材类型": three_d_model.result.get('身材类型', '未知'),
                "腿型": three_d_model.result.get('腿型', '未知'),
            },            
            "face_volume_analysis": volume_result,
            "face_volume_info": {
                "量感分析": volume_result.get('量感分析', '未知'),
                "脸大脸小": volume_result.get('脸大脸小', '未知'),
                "面部留白": volume_result.get('面部留白', '未知'),
                "综合曲直": volume_result.get('综合曲直', '未知'),
                "推荐风格": volume_result.get('推荐风格', '未知'),
            },
            "Face_style": volume_result.get('Face_style', '未知'),
            "body_style": volume_result.get('Final_Curve_Straight', '未知'), #"Straight",
            "body_type": three_d_model.result.get('body_type', '未知'), #"A",
            "leg_type": three_d_model.result.get('leg_type', '未知'), #"O-leg",
        }

        print(result)

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

        
        # 设置 session 的过期时间
        session.permanent = True  # 让 session 在多个请求之间持久化
        app.permanent_session_lifetime = timedelta(minutes=30)  # 设置 session 存活时间 30 分钟

        # Analyze the lip shape
        lip_analyzer = LipShapeAnalyzer(image)
        lip_analyzer.detect_landmarks()
        lip_analyzer.analyze_lip_shape()
        lip_curve = lip_analyzer.result.get('曲直结果', '未知') # 嘴唇曲直结果


        # Analyze the nose shape
        nose_analyzer = NoseAnalyzer(image)
        nose_analyzer.detect_landmarks()
        # 1. Nose width analysis
        nose_analyzer.calculate_distance_ratio()
        # 2. Nose bridge position analysis
        nose_analyzer.analyze_nose_bridge(threshold=20)
        # 3. Nose wing curvature analysis
        nose_analyzer.analyze_nose_wing_curvature()
        # 4. ** The final analysis results **
        nose_analyzer.visualize_nose_bridge_analysis() #***
        nose_curve = nose_analyzer.result.get('鼻型综合曲直', '未知') # 鼻子曲直结果

        # Analyze eye shape
        eye_analyzer = EyeShapeAnalyzer(image)
        eye_analyzer.detect_landmarks()
        eye_analyzer.analyze_eye_shape()
        eye_curve = eye_analyzer.result.get('眼型曲直综合', '未知') # 眼睛曲直结果

        # FaceShapeAnalyze 三庭五眼和气质
        face_analyzer = FaceAnalyzer(image)
        face_analyzer.analyze()
        face_curve = face_analyzer.result.get('脸型曲直', '未知') # 脸型曲直结果  

        # 提取五眼比例
        five_eye_ratio_str = face_analyzer.result.get("五眼比例", "1 : 1 : 1 : 1 : 1")  # 避免数据丢失
        five_eye_ratios = [float(x.strip()) for x in five_eye_ratio_str.split(":")]

        # 计算 ratio_1 和 ratio_5
        ratio_1 = five_eye_ratios[0]  # 第一个数
        ratio_5 = five_eye_ratios[-1]  # 最后一个数

        # 存入 Flask session
        session["ratio_1"] = ratio_1
        session["ratio_5"] = ratio_5
        session["lip_curve"] = lip_curve
        session["nose_curve"] = nose_curve
        session["eye_curve"] = eye_curve
        session["face_curve"] = face_curve
        session.modified = True


        # Collect the analysis results
        result = {
            "lip_shape": lip_analyzer.result, 
            "Lips_detailed_info": {
                "唇形": lip_analyzer.result.get('唇形', '未知'),
                "唇部数据": lip_analyzer.result.get('唇部数据', '未知'),
                '上唇': lip_analyzer.result.get('上唇', '未知'),
                '下唇': lip_analyzer.result.get('下唇', '未知'),
                "上下唇比例": lip_analyzer.result.get('上下唇比例', '未知'),
                "曲直结果": lip_analyzer.result.get('曲直结果', '未知'),            
            },
            "nose_shape": nose_analyzer.result,
            "nose_detailed_info": {
                "鼻翼宽窄判断": nose_analyzer.result.get('鼻翼宽窄', '未知'),
                "右脸山根": nose_analyzer.result.get('右脸山根', '未知'),
                "左脸山根": nose_analyzer.result.get('左脸山根', '未知'),
                "山根曲直": nose_analyzer.result.get('山根曲直', '未知'),
                "鼻翼曲直判断": nose_analyzer.result.get('鼻翼曲线综合判断', '未知'),
                "右鼻翼曲率": nose_analyzer.result.get('右鼻翼曲率', '未知'),
                "左鼻翼曲率": nose_analyzer.result.get('左鼻翼曲率', '未知'),
                "鼻孔比例": nose_analyzer.result.get('鼻孔比例', '未知'),
                "鼻型综合曲直": nose_analyzer.result.get('鼻型综合曲直', '未知'),
            },
            "eye_shape": eye_analyzer.result, 
            "eye_detailed_info": {
                "右眼类型": eye_analyzer.result.get('右眼类型', '未知'),
                "右眼曲直": eye_analyzer.result.get('右眼曲直', '未知'),
                "右眼眼长和眼高的比例": eye_analyzer.result.get('右眼长高比例', '未知'),
                "左眼类型": eye_analyzer.result.get('左眼类型', '未知'),
                "左眼曲直": eye_analyzer.result.get('左眼曲直', '未知'),
                "左眼眼长和眼高的比例": eye_analyzer.result.get('左眼长高比例', '未知'),
                "眼型综合曲直": eye_analyzer.result.get('眼型曲直综合', '未知'),
            },
            "Face_shape": face_analyzer.result, # {'五眼比例': '0.79 : 1 : 1.24 : 0.93 : 0.68', '三庭比例': '1 : 1.63 : 1.46', '三线比例': '0.95 : 1 : 0.88', '脸长和脸宽的比例': '1.3', '下巴形状': '钝弧（圆形下巴）', '脸型判断结果': '圆形脸', '脸部风格': '长中庭, 气质脸'}
            "Face_shape_info": {
                "五眼比例": face_analyzer.result.get('五眼比例', '未知'),
                "三庭比例": face_analyzer.result.get('三庭比例', '未知'),
                "三线比例": face_analyzer.result.get('三线比例', '未知'),
                "脸长和脸宽的比例": face_analyzer.result.get('脸长和脸宽的比例', '未知'),
                "下巴形状": face_analyzer.result.get('下巴形状', '未知'),
                "脸型判断结果": face_analyzer.result.get('脸型判断结果', '未知'),
                "脸部风格": face_analyzer.result.get('脸部风格', '未知'),
                "脸型曲直": face_analyzer.result.get('脸型曲直', '未知'),
            },
            "Face_shape_type": face_analyzer.result.get('七种脸型分类', '未知'), #"Egg",
            

            #  "body_shape_info": body_analyzer.result
        }

        print("Session after face-analyze:", dict(session))  # 打印 session 数据

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
    app.run(host='0.0.0.0', port=5000)  #debug = True
