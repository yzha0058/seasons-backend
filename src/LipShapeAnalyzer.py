import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

class LipShapeAnalyzer:
    def __init__(self, image):
        self.image = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.landmarks_coords = {}
        self.result = ''
        self.final_shape = ''
        self.mouth_orientation = ''
        self.right_angle = 0
        self.left_angle = 0

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

    @staticmethod
    def calculate_angle(p1, p2, p3):
        vec1 = np.array(p1) - np.array(p2)
        vec2 = np.array(p3) - np.array(p2)
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    def judge_round_wide_lip(self):
        pupil_distance = np.linalg.norm(np.array(self.landmarks_coords[469]) - np.array(self.landmarks_coords[476]))
        lip_width = np.linalg.norm(np.array(self.landmarks_coords[61]) - np.array(self.landmarks_coords[291]))
        if lip_width <= pupil_distance:
            return 'A', '圆唇'  # 圆唇
        else:
            return 'B', '宽唇'  # 宽唇

    def judge_thin_thick_lip(self):
        lip_width = np.linalg.norm(np.array(self.landmarks_coords[61]) - np.array(self.landmarks_coords[291]))
        midpoint_37_267 = (
            (self.landmarks_coords[37][0] + self.landmarks_coords[267][0]) / 2,
            (self.landmarks_coords[37][1] + self.landmarks_coords[267][1]) / 2
        )
        lip_thickness = np.linalg.norm(np.array(midpoint_37_267) - np.array(self.landmarks_coords[17]))
        
        if lip_thickness < lip_width / 2:
            return 'A', '偏薄'  # 偏薄
        else:
            return 'B', '偏厚'  # 偏厚

    def judge_m_lip(self):
        m_lip_angle = self.calculate_angle(
            self.landmarks_coords[37], self.landmarks_coords[0], self.landmarks_coords[267]
        )
        if m_lip_angle >= 20:
            return 'A', 'M型唇'  # M型唇
        else:
            return 'B', '无明显唇峰'  # 无明显唇峰

    def judge_mouth_orientation(self):
        lip_line_points = [
            self.landmarks_coords[80], self.landmarks_coords[81], self.landmarks_coords[82],
            self.landmarks_coords[13], self.landmarks_coords[312], self.landmarks_coords[311], self.landmarks_coords[310]
        ]
        m_lip, b_lip = self.fit_line(lip_line_points)
        m_right, b_right = self.fit_line([self.landmarks_coords[61], self.landmarks_coords[78]])
        m_left, b_left = self.fit_line([self.landmarks_coords[308], self.landmarks_coords[291]])

        self.right_angle = self.calculate_angle(
            self.landmarks_coords[61], self.landmarks_coords[78], self.landmarks_coords[291]
        )
        self.left_angle = self.calculate_angle(
            self.landmarks_coords[308], self.landmarks_coords[291], self.landmarks_coords[61]
        )

        if self.right_angle > 20 and self.left_angle > 20:
            if self.landmarks_coords[61][1] < m_lip * self.landmarks_coords[61][0] + b_lip and \
               self.landmarks_coords[308][1] < m_lip * self.landmarks_coords[308][0] + b_lip:
                return 'B', '嘴角上扬'  # 嘴角上扬
            elif self.landmarks_coords[61][1] > m_lip * self.landmarks_coords[61][0] + b_lip and \
                 self.landmarks_coords[308][1] > m_lip * self.landmarks_coords[308][0] + b_lip:
                return 'A', '嘴角下垂'  # 嘴角下垂
        return 'B', '嘴角平坦'  # 嘴角平坦

    @staticmethod
    def fit_line(points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        m, b = np.polyfit(x, y, 1)
        return m, b

    def analyze_lip_shape(self):
        shape1, round_wide_desc = self.judge_round_wide_lip()
        shape2, thin_thick_desc = self.judge_thin_thick_lip()
        shape3, m_lip_desc = self.judge_m_lip()
        shape4, mouth_orientation_desc = self.judge_mouth_orientation()

        self.final_shape = shape1 + shape2 + shape3 + shape4

        # 根据 final_shape 进行结果判断
        if self.final_shape == 'AAAB':
            self.result = '性感M唇'
        elif self.final_shape == 'ABAB':
            self.result = '甜美微笑唇'
        elif self.final_shape == 'BAAB':
            self.result = '黄金比例标准唇'
        elif self.final_shape == 'AABB':
            self.result = '叶形唇'
        elif self.final_shape == 'BABB':
            self.result = '薄唇'
        elif 'A' in self.final_shape[0] or 'A' in self.final_shape[3]:
            self.result = '覆舟唇'
        elif 'B' in self.final_shape[1] and 'B' in self.final_shape[3]:
            self.result = '厚唇'
        else:
            self.result = '未知类型'

        # 输出详细信息
        print(f"唇形：{round_wide_desc}，{thin_thick_desc}，{m_lip_desc}")
        print(f"嘴角：{mouth_orientation_desc}，右嘴角倾斜度：{self.right_angle:.2f}°，左嘴角倾斜度：{self.left_angle:.2f}°")