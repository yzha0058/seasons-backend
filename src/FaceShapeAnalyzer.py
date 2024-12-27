import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import matplotlib.pyplot as plt

class FaceAnalyzer:
    def __init__(self, image):
        self.image = image
        self.image_path = image
        self.image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.landmarks = None  # 存储提取到的关键点
        self.result = {}  # 用于存储分析结果

        # 初始化YOLO模型
        self.yolo_model = YOLO('yolov8n-face.pt')

    def detect_face_with_yolo(self):
        """使用YOLO检测人脸"""
        results = self.yolo_model(self.image)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        return results

    def extract_landmarks_with_mediapipe(self):
        """使用MediaPipe提取人脸关键点"""
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            results = face_mesh.process(self.image_rgb)
            if results.multi_face_landmarks:
                self.landmarks = results.multi_face_landmarks[0].landmark
                return True
            else:
                print("No landmarks detected.")
                return False

    def get_point(self, index):
        """获取关键点的像素坐标"""
        x = int(self.landmarks[index].x * self.image.shape[1])
        y = int(self.landmarks[index].y * self.image.shape[0])
        return np.array([x, y])

    def calculate_five_eyes(self):
        """计算五眼比例"""
        x_127 = self.get_point(127)[0]
        x_356 = self.get_point(356)[0]
        x_130 = self.get_point(130)[0]
        x_133 = self.get_point(133)[0]
        x_362 = self.get_point(362)[0]
        x_359 = self.get_point(359)[0]

        region_1_width = x_130 - x_127
        region_2_width = x_133 - x_130
        region_3_width = x_362 - x_133
        region_4_width = x_359 - x_362
        region_5_width = x_356 - x_359

        if region_2_width == 0:
            raise ValueError("Region width is zero, cannot calculate ratios.")

        ratios = [
            round(region_1_width / region_2_width, 2),
            1,
            1,
            round(region_4_width / region_2_width, 2),
            round(region_5_width / region_2_width, 2),
        ]
        self.result["五眼比例"] = f"{ratios[0]} : {ratios[1]} : {ratios[2]} : {ratios[3]} : {ratios[4]}"
        return ratios

    def calculate_three_ratios(self):
        """计算三庭比例"""
        y_keypoint_9 = int(self.landmarks[9].y * self.image.shape[0])
        y_keypoint_2 = int(self.landmarks[2].y * self.image.shape[0])

        # 假设人脸框的顶部和底部
        y_top = 0
        y_bottom = self.image.shape[0]

        region_1_height = y_keypoint_9 - y_top
        region_2_height = y_keypoint_2 - y_keypoint_9
        region_3_height = y_bottom - y_keypoint_2

        if region_1_height == 0:
            raise ValueError("Top region height is zero, cannot calculate ratios.")

        ratios = [
            1,
            round(region_2_height / region_1_height, 2),
            round(region_3_height / region_1_height, 2),
        ]
        self.result["三庭比例"] = f"{ratios[0]} : {ratios[1]} : {ratios[2]}"
        return ratios

    def calculate_face_ratios(self):
        """计算三线比例和脸长脸宽比例"""
        # 1. 计算额头宽度 (21, 251)
        point_21 = self.get_point(21)
        point_251 = self.get_point(251)
        forehead_width = np.linalg.norm(point_21 - point_251)

        # 2. 计算颧骨宽度 (234, 454)
        point_234 = self.get_point(234)
        point_454 = self.get_point(454)
        cheekbone_width = np.linalg.norm(point_234 - point_454)

        # 3. 计算下颌宽度 (58, 288)
        point_58 = self.get_point(58)
        point_288 = self.get_point(288)
        jaw_width = np.linalg.norm(point_58 - point_288)

        # 4. 计算脸长 (检测框顶部中点到关键点 152)
        y_top = 0
        x_center = self.image.shape[1] // 2
        point_top_center = np.array([x_center, y_top])
        point_152 = self.get_point(152)
        face_length = np.linalg.norm(point_top_center - point_152)

        # 计算比例
        forehead_ratio = round(forehead_width / cheekbone_width, 2)
        jaw_ratio = round(jaw_width / cheekbone_width, 2)
        face_length_ratio = round(face_length / cheekbone_width, 2)

        # 打印结果
        print(f"三线比例 (额头: 颧骨 : 下颌) = {forehead_ratio} : 1 : {jaw_ratio}")
        print(f"脸长和脸宽的比例: {face_length_ratio}")
        self.result["三线比例 "] = f"{forehead_ratio} : 1 : {jaw_ratio}"
        self.result["脸长和脸宽的比例 "] = f"{face_length_ratio}"


        # 可视化关键点和计算结果
        cv2.line(self.image_rgb, tuple(point_21), tuple(point_251), (255, 0, 0), 2)  # 额头宽度
        cv2.line(self.image_rgb, tuple(point_234), tuple(point_454), (0, 255, 0), 2)  # 颧骨宽度
        cv2.line(self.image_rgb, tuple(point_58), tuple(point_288), (0, 0, 255), 2)  # 下颌宽度
        cv2.line(self.image_rgb, tuple(point_top_center), tuple(point_152), (255, 255, 0), 2)  # 脸长
        # 返回脸长脸宽比例
        return face_length_ratio
    




    def analyze_chin_shape(self):
        """分析下巴形状"""
        chin_points_indices = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]
        chin_points = [self.get_point(index) for index in chin_points_indices]

        # 计算下巴长度和下颌宽度
        point_58 = self.get_point(58)
        point_288 = self.get_point(288)
        midpoint_58_288 = (point_58 + point_288) // 2
        point_152 = self.get_point(152)

        chin_length = np.linalg.norm(midpoint_58_288 - point_152)
        chin_width = np.linalg.norm(point_58 - point_288)
        chin_length_width_ratio = chin_width / chin_length

        # 判断下巴形状
        if chin_length_width_ratio < 2.1:
            chin_type = "锐弧（尖形下巴）"
        elif 2.1 <= chin_length_width_ratio <= 2.6:
            chin_type = "钝弧（圆形下巴）"
        else:
            chin_type = "机器人下巴（方形下巴）"

        print(f"下巴形状: {chin_type}")
        print(f"下巴宽长比: {chin_length_width_ratio:.2f}")
        self.result["下巴形状 "] = f"{chin_type}"

        # 在图像上绘制下巴关键点连线和下巴形状
        for i in range(len(chin_points) - 1):
            cv2.line(self.image_rgb, tuple(chin_points[i]), tuple(chin_points[i + 1]), (0, 255, 0), 2)

    def determine_face_shape(self):
        """判断脸型"""
        # 计算所需的比例
        forehead_width = np.linalg.norm(self.get_point(21) - self.get_point(251))
        cheekbone_width = np.linalg.norm(self.get_point(234) - self.get_point(454))
        jaw_width = np.linalg.norm(self.get_point(58) - self.get_point(288))
        face_length_ratio = self.calculate_face_ratios()

        # 判断下巴形状
        chin_type = self.analyze_chin_shape()

        # 判断脸型
        face_shape = ""
        if (forehead_width > cheekbone_width > jaw_width) and (1.4 <= face_length_ratio <= 1.6) and (chin_type != "机器人下巴（方形下巴）"):
            face_shape = "椭圆脸（鹅蛋脸）"
        elif (abs(forehead_width - jaw_width) < 0.05 * forehead_width) and (face_length_ratio < 1.4) and (chin_type == "钝弧（圆形下巴）"):
            face_shape = "圆形脸"
        elif (abs(forehead_width - cheekbone_width) < 0.05 * cheekbone_width) and (abs(cheekbone_width - jaw_width) < 0.05 * jaw_width):
            if jaw_width >= cheekbone_width and chin_type == "锐弧（尖形下巴）":
                face_shape = "方形脸（锐弧）"
            elif jaw_width >= cheekbone_width and chin_type == "钝弧（圆形下巴）":
                face_shape = "方形脸（钝弧）"
            else:
                face_shape = "方形脸"
        elif (cheekbone_width > forehead_width and cheekbone_width > jaw_width):
            face_shape = "菱形脸"
        elif (forehead_width > cheekbone_width > jaw_width):
            face_shape = "倒三角脸"
        elif (cheekbone_width <= jaw_width):
            face_shape = "梨形脸"

        print(f"脸型判断结果: {face_shape}")

        self.result["脸型判断结果 "] = f"{face_shape}"



    def determine_three_ratios_type(self):
        """计算三庭比例类型"""
        ratios = self.calculate_three_ratios()
        three_ratios_type = ""
        tolerance = 0.1

        if abs(ratios[0] - 1) < tolerance and abs(ratios[1] - 1) < tolerance and abs(ratios[2] - 1) < tolerance:
            three_ratios_type = "标准脸"
        elif abs(ratios[0] - 1) < tolerance and abs(ratios[1] - 1) < tolerance and abs(ratios[2] - 0.8) < tolerance:
            three_ratios_type = "幼态脸"
        elif abs(ratios[0] - 1) < tolerance and abs(ratios[1] - 1.2) < tolerance and abs(ratios[2] - 1) < tolerance:
            three_ratios_type = "御姐成熟脸"
        elif abs(ratios[0] - 1) < tolerance and abs(ratios[1] - 1.2) < tolerance and abs(ratios[2] - 1.5) < tolerance:
            three_ratios_type = "高气场/明艳脸"

        if three_ratios_type:
            print(f"三庭比例类型: {three_ratios_type}")

        self.result["气质类型 "] = f"{three_ratios_type}"

    def analyze(self):
        """主分析流程"""
        self.detect_face_with_yolo()
        if not self.extract_landmarks_with_mediapipe():
            return

        self.calculate_five_eyes()
        self.calculate_three_ratios()
        self.calculate_face_ratios()
        self.analyze_chin_shape()
        self.determine_face_shape()
        self.determine_three_ratios_type()

    def visualize_results(self):
        """可视化结果"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image_rgb)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()

# 初始化分析器
# image_path = "images/SCUT-FBP-106.jpg"
# face_analyzer = FaceAnalyzer(image_path)
# face_analyzer.analyze()
# face_analyzer.visualize_results()