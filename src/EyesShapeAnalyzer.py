from PIL import Image, ImageDraw, ImageFont
import cv2
import mediapipe as mp
import numpy as np
import os


class EyeShapeAnalyzer:
    def __init__(self, image):
        self.image = image
        # self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.landmarks_coords = {}
        self.result = ''

    def detect_landmarks(self):
        mp_face_landmarker = mp.solutions.face_mesh
        face_landmarker = mp_face_landmarker.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        
        results = face_landmarker.process(self.image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * self.image.shape[1])
                    y = int(landmark.y * self.image.shape[0])
                    self.landmarks_coords[idx] = (x, y)

    # 计算三点构成的角度其中P2是顶点
    def calculate_angle(self, p1, p2, p3):
        vec1 = np.array(p2) - np.array(p1)
        vec2 = np.array(p3) - np.array(p1)
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    
    def rotate_point(self, point, center, angle):
        # 旋转某点绕中心点的指定角度（角度制）
        angle_rad = np.radians(angle)
        x, y = point[0] - center[0], point[1] - center[1]
        new_x = x * np.cos(angle_rad) - y * np.sin(angle_rad)
        new_y = x * np.sin(angle_rad) + y * np.cos(angle_rad)
        return new_x, new_y


    def analyze_eye_shape(self):  
        #以图片左上角为原点 Zhoujy
        # Step 1: 眼尾高低   
        right_eye_outer_inner = 'A' if self.landmarks_coords[33][1] < self.landmarks_coords[154][1] else 'B'
        left_eye_outer_inner = 'A' if self.landmarks_coords[263][1] < self.landmarks_coords[362][1] else 'B'

        # Step 2: 内眼角角度 [153 154 158]  [362 384 381]
        right_inner_angle = self.calculate_angle(self.landmarks_coords[154], self.landmarks_coords[158], self.landmarks_coords[153])
        left_inner_angle = self.calculate_angle(self.landmarks_coords[362], self.landmarks_coords[384], self.landmarks_coords[381])
        right_eye_inner_angle = 'A' if right_inner_angle < 45 else 'B'
        left_eye_inner_angle = 'A' if left_inner_angle < 45 else 'B'

        # Step 3: 眼长眼高比例  
        right_eye_length = np.linalg.norm(np.array(self.landmarks_coords[33]) - np.array(self.landmarks_coords[154]))
        right_eye_height = np.linalg.norm(np.array(self.landmarks_coords[159]) - np.array(self.landmarks_coords[145]))
        right_eye_length_height = 'A' if right_eye_length <= 2 * right_eye_height else 'B'

        left_eye_length = np.linalg.norm(np.array(self.landmarks_coords[263]) - np.array(self.landmarks_coords[381]))
        left_eye_height = np.linalg.norm(np.array(self.landmarks_coords[386]) - np.array(self.landmarks_coords[374]))
        left_eye_length_height = 'A' if left_eye_length <= 2 * left_eye_height else 'B'

        # Step 4: 使用瞳孔连线作为新的x轴判断外眼角上扬或下垂
        right_pupil = self.landmarks_coords[468]
        left_pupil = self.landmarks_coords[473]
        outer_right_eye = self.landmarks_coords[33]
        outer_left_eye = self.landmarks_coords[263]

        # 计算瞳孔中心连线的角度
        dx = left_pupil[0] - right_pupil[0]
        dy = left_pupil[1] - right_pupil[1]
        angle = np.degrees(np.arctan2(dy, dx))

        # 将外眼角点旋转到新坐标系
        new_right_outer = self.rotate_point(outer_right_eye, right_pupil, -angle)
        new_left_outer = self.rotate_point(outer_left_eye, left_pupil, -angle)

        # 判断外眼角相对于瞳孔连线的y坐标位置
        right_eye_outer_tilt = 'A' if new_right_outer[1] < 0 else 'B'
        left_eye_outer_tilt = 'A' if new_left_outer[1] < 0 else 'B'

        # 综合结果
        
        right_eye_shape = right_eye_outer_inner + right_eye_inner_angle + right_eye_length_height + right_eye_outer_tilt
        left_eye_shape = left_eye_outer_inner + left_eye_inner_angle + left_eye_length_height + left_eye_outer_tilt

        # 获取眼型和特征 
        right_eye_type = self.get_eye_type(right_eye_shape, right_inner_angle, right_eye_length, right_eye_height)
        left_eye_type = self.get_eye_type(left_eye_shape, left_inner_angle, left_eye_length, left_eye_height)

        # 输出结果
        print(f"右眼形状代码: {right_eye_shape} -> 眼型: {right_eye_type[0]}，特征: {right_eye_type[1]}，内眼角角度: {right_inner_angle:.2f}°，眼长和眼高的比例：{right_eye_length/right_eye_height:.2f}")
        print(f"左眼形状代码: {left_eye_shape} -> 眼型: {left_eye_type[0]}，特征: {left_eye_type[1]}，内眼角角度: {left_inner_angle:.2f}°,眼长和眼高的比例：{left_eye_length/left_eye_height:.2f}")
        
        # 保存分析结果
        self.result = {
            "右眼形状": right_eye_type[0],
            "右眼内眼角角度": f"{right_inner_angle:.2f}°",
            "右眼眼长和眼高的比例": f"{right_eye_length / right_eye_height:.2f}",
            "左眼形状": left_eye_type[0],
            "左眼内眼角角度": f"{left_inner_angle:.2f}°",
            "左眼眼长和眼高的比例": f"{left_eye_length / left_eye_height:.2f}"
        }
        return right_eye_shape, right_eye_type[0], left_eye_shape, left_eye_type[0]

    
    def get_eye_type(self, eye_shape, inner_angle, eye_length, eye_height): #最常出现的是 ABBA ABBB
        # 眼型判断规则
        eye_shapes = {  # 杏眼、圆眼(0.8)、桃花眼、丹凤眼、 下垂眼、细长眼、柳叶眼
            '杏眼': ['ABAA', 'ABAB', 'AAAB'],
            '桃花眼': ['AAAA', 'BAAA'],
            '丹凤眼': ['ABBA', 'AABA'],
            '圆眼': ['BBAB'],
            '下垂眼': ['BABB', 'BBAA'],
            '细长眼': ['BBBB', 'ABBB', 'BBBA'],  # 大于2倍 
            '柳叶眼': ['BABA', 'BAAB', 'AABB']
        }
        for eye_type, codes in eye_shapes.items():
            if eye_shape in codes:
                # 判断偏圆润还是细长
                if eye_length <= 1.5 * eye_height:
                    round_long = '圆润'
                else:
                    round_long = '细长'
                return eye_type, round_long
        return '未知类型', '未知形状'
   
    def display_result(self):
        self.analyze_eye_shape()
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
