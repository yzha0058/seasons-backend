import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

class LipShapeAnalyzer:
    def __init__(self, image):
        self.image = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.landmarks_coords = {}
        self.result ={}
        self.final_shape = ''
        self.mouth_orientation = ''
        self.right_angle = 0
        self.left_angle = 0
        self.m_lip_angle = 0
        self.lower_lip_upper_lip_ratio = 0

    
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
        # 计算三点的夹角 其中p2 是顶点
        vec1 = np.array(p1) - np.array(p2)
        vec2 = np.array(p3) - np.array(p2)
        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    @staticmethod
    def calculate_line_angle(m1, m2):
    # 计算两条直线斜率之间的夹角，并限制在 0° 到 90° 范围内
        theta = np.arctan(abs((m1 - m2) / (1 + m1 * m2)))
        return np.degrees(theta)

    #判断圆唇和宽唇
    def judge_round_wide_lip(self):
        pupil_distance = np.linalg.norm(np.array(self.landmarks_coords[469]) - np.array(self.landmarks_coords[476])) #内瞳距
        lip_width = np.linalg.norm(np.array(self.landmarks_coords[61]) - np.array(self.landmarks_coords[291])) #唇宽
        if lip_width <= pupil_distance:
            return 'A', '圆唇'  # 圆唇  偏曲
        else:
            return 'B', '宽唇'  # 宽唇  偏直
            
    #判断薄唇和厚唇
    def judge_thin_thick_lip(self):
        lip_width = np.linalg.norm(np.array(self.landmarks_coords[78]) - np.array(self.landmarks_coords[308]))
        midpoint_37_267 = (
            (self.landmarks_coords[37][0] + self.landmarks_coords[267][0]) / 2,
            (self.landmarks_coords[37][1] + self.landmarks_coords[267][1]) / 2
        )
        lip_thickness = np.linalg.norm(np.array(midpoint_37_267) - np.array(self.landmarks_coords[17]))
        
        if lip_thickness < lip_width / 2:
            return 'A', '偏薄'  # 偏薄   偏直
        else:
            return 'B', '偏厚'  # 偏厚   偏曲

    #判断唇峰
    def judge_m_lip(self):
        self.m_lip_angle = 180 - self.calculate_angle(
            self.landmarks_coords[37], self.landmarks_coords[0], self.landmarks_coords[267]
        )
        if self.m_lip_angle  >= 20:
            return 'A', 'M型唇峰'  # M型唇  偏直
        else:
            return 'B', '无明显唇峰'  # 无明显唇峰   偏曲

    #判断左右嘴角的角度 上扬还是下垂
    def judge_mouth_orientation(self):
        # 嘴唇中心线拟合
        lip_line_points = [
            self.landmarks_coords[80], self.landmarks_coords[81], self.landmarks_coords[82],
            self.landmarks_coords[13], self.landmarks_coords[312], self.landmarks_coords[311], self.landmarks_coords[310]
        ]
        m_lip, b_lip = self.fit_line(lip_line_points)

        # 右嘴角的拟合线和夹角
        m_right, _ = self.fit_line([self.landmarks_coords[61], self.landmarks_coords[78]])
        self.right_angle = self.calculate_line_angle(m_right, m_lip)

        # 左嘴角的拟合线和夹角
        m_left, _ = self.fit_line([self.landmarks_coords[308], self.landmarks_coords[291]])
        self.left_angle = self.calculate_line_angle(m_left, m_lip)

        # 找出较大的角度
        if self.right_angle > self.left_angle:
            max_angle = self.right_angle
            point = self.landmarks_coords[61]
            side = '右'
        else:
            max_angle = self.left_angle
            point = self.landmarks_coords[308]
            side = '左'
    
        # 判断嘴角朝向|
        if max_angle > 15:
            if point[1] < m_lip * point[0] + b_lip:
                return 'B', '嘴角上扬'  # f'{side}嘴角上扬   偏曲
            else:
                return 'A', '嘴角下垂'  # f'{side}嘴角下垂   偏直
        else:
            return 'B', '嘴角平坦'  # 嘴角平坦

    @staticmethod
    def fit_line(points):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        m, b = np.polyfit(x, y, 1)
        return m, b

    
    # NEW判断上下唇厚度偏薄偏厚及偏直偏曲
    def judge_upper_lower_lip_thickness(self):
        # 上唇厚度：distance(0,13)
        upper_lip_thickness = np.linalg.norm(np.array(self.landmarks_coords[0]) - np.array(self.landmarks_coords[13]))
        # 鼻底到唇峰距离：distance(94,13)
        nose_to_lippeak = np.linalg.norm(np.array(self.landmarks_coords[94]) - np.array(self.landmarks_coords[13]))

        # 判断上唇
        # 如果 upper_lip_thickness < nose_to_lippeak/3 => 偏薄偏直，否则偏厚偏曲
        if upper_lip_thickness < (nose_to_lippeak/3):
            upper_desc = "偏薄偏直"
        else:
            upper_desc = "偏厚偏曲"

        # 下唇厚度：distance(14,17)
        lower_lip_thickness = np.linalg.norm(np.array(self.landmarks_coords[14]) - np.array(self.landmarks_coords[17]))
        lower_lip_upper_lip_ratio = lower_lip_thickness / upper_lip_thickness
        self.lower_lip_upper_lip_ratio = lower_lip_upper_lip_ratio
        
        print(f"上下唇比例为{lower_lip_upper_lip_ratio}")
        self.result["上下唇比例"] = lower_lip_upper_lip_ratio
        # 如果下唇厚度 > 上唇厚度*1.5 => 偏厚偏曲，否则偏薄偏直
        if lower_lip_upper_lip_ratio > 1.6:
            lower_desc = "偏厚偏曲"
        else:
            lower_desc = "偏薄偏直"
        
        return upper_desc, lower_desc

    
    def analyze_lip_shape(self):
        shape1, round_wide_desc = self.judge_round_wide_lip()
        shape2, thin_thick_desc = self.judge_thin_thick_lip()
        shape3, m_lip_desc = self.judge_m_lip()
        shape4, mouth_orientation_desc = self.judge_mouth_orientation()

        self.final_shape = shape1 + shape2 + shape3 + shape4

        # 根据 final_shape 进行结果判断
        if self.final_shape == 'AAAB':
            self.result["唇形"] = "性感M唇"
        elif self.final_shape == 'ABAB':
            self.result["唇形"] = "甜美微笑唇"
        elif self.final_shape == 'BAAB':
            self.result["唇形"] = "黄金比例标准唇"
        elif self.final_shape == 'AABB':
            self.result["唇形"] = "叶形唇"
        elif self.final_shape == 'BABB':
            self.result["唇形"] = "薄唇"
        elif 'A' in self.final_shape[3]:
            self.result["唇形"] = "覆舟唇"
        elif 'B' in self.final_shape[1] and 'B' in self.final_shape[3]:
            self.result["唇形"] = "厚唇"
        else:
            self.result["唇形"] = "未知类型"

        # 输出详细信息
        self.result["唇部数据"] = f"{self.judge_round_wide_lip()[1]}, {self.judge_thin_thick_lip()[1]}, {self.judge_m_lip()[1]}"
        self.result["嘴角"] = f"{self.judge_mouth_orientation()[1]}, 右嘴角倾斜度:{self.right_angle:.2f}°, 左嘴角倾斜度:{self.left_angle:.2f}°"
    
        # 调用新增方法判断上下唇厚度属性
        upper_desc, lower_desc = self.judge_upper_lower_lip_thickness()
        print(f"上唇：{upper_desc}, 下唇：{lower_desc}")
        self.result["上唇"] = f"{upper_desc}"
        self.result["下唇"] = f"{lower_desc}"
        
        # 根据规则统计偏直与偏曲出现次数
        # round_wide_lip: A=圆唇=偏曲, B=宽唇=偏直
        # thin_thick_lip: A=偏薄=偏直, B=偏厚=偏曲
        # m_lip: A= M型唇=偏直, B=无明显唇峰=偏曲
        # mouth_orientation: B=偏曲, A=偏直
        # 上下唇厚度新判定：包含"偏直"或"偏曲"关键字
        def is_straight(desc):
            return "偏直" in desc
        def is_curved(desc):
            return "偏曲" in desc

        # 特征结果收集
        features_desc = []
        # round_wide_desc: 圆唇=偏曲, 宽唇=偏直
        features_desc.append("偏曲" if shape1=='A' else "偏直")
        # thin_thick_desc: A=偏薄=偏直, B=偏厚=偏曲
        features_desc.append("偏直" if shape2=='A' else "偏曲")
        # m_lip_desc: A= M型唇=偏直, B=无明显唇峰=偏曲
        features_desc.append("偏直" if shape3=='A' else "偏曲")
        # mouth_orientation: B=偏曲, A=偏直
        features_desc.append("偏曲" if shape4=='B' else "偏直")

        # 加入上、下唇厚度描述
        # 这里upper_desc和lower_desc中都包含"偏薄"/"偏厚"和"偏直"/"偏曲"
        # 我们只需要关注"偏直"/"偏曲"
        features_desc.append("偏直" if is_straight(upper_desc) else "偏曲")
        features_desc.append("偏直" if is_straight(lower_desc) else "偏曲")
        print(features_desc)
        #统计偏直与偏曲
        straight_count = features_desc.count("偏直")
        curved_count = features_desc.count("偏曲")
        
        if curved_count >= straight_count:
            overall = "偏曲"
        else:
            overall = "偏直"
        
        print(f"曲直结果：{overall}")
        self.result["曲直结果"] = overall
    
    def display_result(self):
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    
    def get_result_text(self):
        print(f" {self.result}\n")
        return f"唇形: {self.result}\n分类代码: {self.final_shape}\n唇峰: {self.m_lip_angle}°"