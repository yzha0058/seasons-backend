import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt


class BodyShapeAnalyzer:
    def __init__(self, image):
        self.image = image
        # self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.landmarks_coords = {}
        self.result = {}  # 用于存储分析结果

    def detect_landmarks(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)
        results = pose.process(self.image_rgb)

        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                self.landmarks_coords[idx] = (landmark.x, landmark.y)
            self.pose_landmarks = results.pose_landmarks
            return True
        else:
            self.pose_landmarks = None
            return False

    def calculate_head_shoulder_ratio(self):
        # 计算头高
        point4 = self.landmarks_coords[4]
        point10 = self.landmarks_coords[10]
        point1 = self.landmarks_coords[1]
        point9 = self.landmarks_coords[9]
        head_height = max(
            np.linalg.norm(np.array(point4) - np.array(point10)),
            np.linalg.norm(np.array(point1) - np.array(point9))
        ) * 2

        # 计算肩宽
        left_shoulder = self.landmarks_coords[11]
        right_shoulder = self.landmarks_coords[12]
        shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))

        # 计算头肩比
        shoulder_ratio = shoulder_width / head_height

        # 判断头肩比类型
        if shoulder_ratio < 1.5:
            shoulder_type = "肩偏窄"
        elif shoulder_ratio > 1.7:
            shoulder_type = "肩偏宽"
        else:
            shoulder_type = "肩正常"

        self.result["头肩比"] = round(shoulder_ratio, 2)
        self.result["头肩比判断"] = shoulder_type

    def calculate_body_proportion(self):
        # 计算上半身长度（肩膀中点到臀部中点）
        left_shoulder = self.landmarks_coords[11]
        right_shoulder = self.landmarks_coords[12]
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)

        left_hip = self.landmarks_coords[23]
        right_hip = self.landmarks_coords[24]
        mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)
        upper_body_length = np.linalg.norm(np.array(mid_shoulder) - np.array(mid_hip))

        # 计算下半身长度（臀部中点到脚踝中点）
        left_ankle = self.landmarks_coords[31]
        right_ankle = self.landmarks_coords[32]
        mid_ankle = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)
        lower_body_length = np.linalg.norm(np.array(mid_hip) - np.array(mid_ankle))

        # 标准化上下半身比例
        head_height = max(
            np.linalg.norm(np.array(self.landmarks_coords[4]) - np.array(self.landmarks_coords[10])),
            np.linalg.norm(np.array(self.landmarks_coords[1]) - np.array(self.landmarks_coords[9]))
        ) * 2
        upper_body_ratio = upper_body_length / head_height
        lower_body_ratio = lower_body_length / head_height
        body_ratio = lower_body_ratio / upper_body_ratio

        # 判断比例类型
        if body_ratio > 1:
            if 1 <= body_ratio <= 1.25:
                proportion_type = "五五身"
            elif 1.26 <= body_ratio <= 1.55:
                proportion_type = "四六身（显高）"
            elif body_ratio > 1.56:
                proportion_type = "三七身（黄金比例）"
        else:
            proportion_type = "六四分（显矮）"

        self.result["上下半身比例"] = round(body_ratio, 2)
        self.result["比例判断"] = proportion_type

    def display_result(self):
        if self.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            mp_drawing_styles = mp.solutions.drawing_styles
            annotated_image = self.image.copy()
            mp_drawing.draw_landmarks(
                annotated_image,
                self.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )
            plt.figure(figsize=(12, 12))
            plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.title("Pose Detection with Standardized Measurements")
            plt.show()
