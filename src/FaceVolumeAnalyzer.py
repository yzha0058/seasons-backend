import cv2
import numpy as np
from flask import session
from BodyShapeAnalyzer import PoseAnalyzer

class FaceVolumeAnalyzer:
    def __init__(self, image, height_cm, lip_curve, nose_curve, eye_curve, face_curve, ratio_1, ratio_5):
        """ 前端输入的数据进行初始化 """
        self.image = image
        self.height_cm = height_cm  # 用户输入的身高
        self.lip_curve = lip_curve  # 唇型曲直
        self.nose_curve = nose_curve  # 鼻型曲直    
        self.eye_curve = eye_curve  # 眼型曲直
        self.face_curve = face_curve  # 脸型曲直
        self.ratio_1 = ratio_1  # 五眼比例1
        self.ratio_5 = ratio_5  # 五眼比例5
        self.result = {}
    
    def analyze(self):
        # 运行姿态分析
        body_analyzer = PoseAnalyzer(self.image)

        
        # 获取 Pose 关键点
        landmarks = body_analyzer.landmarks
        
        # 计算头长
        head_height = np.linalg.norm(
            np.array([(landmarks[7].x + landmarks[8].x) / 2, (landmarks[7].y + landmarks[8].y) / 2]) -
            np.array([(landmarks[9].x + landmarks[10].x) / 2, (landmarks[9].y + landmarks[10].y) / 2])
        ) * 3
        
        # 计算身高
        body_height = np.linalg.norm(
            np.array([(landmarks[7].x + landmarks[8].x) / 2, (landmarks[7].y + landmarks[8].y) / 2]) -
            np.array([(landmarks[29].x + landmarks[30].x) / 2, (landmarks[29].y + landmarks[30].y) / 2])
        ) + head_height
        
        # 计算头身比例
        head_ratio = head_height / body_height
        
        # 计算头宽肩宽比
        head_width = np.linalg.norm(np.array([landmarks[7].x, landmarks[7].y]) - np.array([landmarks[8].x, landmarks[8].y]))
        shoulder_width = np.linalg.norm(np.array([landmarks[11].x, landmarks[11].y]) - np.array([landmarks[12].x, landmarks[12].y]))
        head_shoulder_ratio = head_width / shoulder_width
        

        # 计算面部留白
        face_whitespace_ratio = (self.ratio_1 + self.ratio_5) / 2
        face_whitespace = "面部留白大" if face_whitespace_ratio > 1 else "面部留白小"
        
        # 计算大脸还是小脸
        small_face = (head_ratio <= 0.125) + (head_shoulder_ratio <= 0.66)
        large_face = (head_ratio >= 0.166) + (head_shoulder_ratio >= 0.75)

        if small_face >= 2:
            face_size = "小脸"
        elif large_face >= 2:
            face_size = "大脸"
        else:
            face_size = "正常大小"

        # 量感判断
        if face_whitespace == "面部留白大" and face_size == "小脸" and self.height_cm < 158:
            face_volume = "量感小"
        elif face_whitespace == "面部留白小" and face_size == "大脸" and self.height_cm > 168:
            face_volume = "量感大"
        else:
            face_volume = "中量感"


        # 计算综合曲直得分
        curve_score = {"偏曲": 1, "偏直": -1, "曲直适中": 0, "未知": 0}
        total_curve_score = (curve_score[self.eye_curve] * 0.3 +
                             curve_score[self.lip_curve] * 0.2 +
                             curve_score[self.face_curve] * 0.25 +
                             curve_score[self.nose_curve] * 0.25)        
        
        # 风格判断
        if face_volume == "量感小":
            style = "少女" if total_curve_score > 0 else "少年"
        elif face_volume == "量感大":
            style = "浪漫" if total_curve_score > 0 else "古典/戏剧"
        else:
            if total_curve_score > 0:
                style = "优雅自然"
            elif total_curve_score < 0:
                style = "前卫"
            else:
                style = "自然"
        
        self.result["脸大脸小"] = face_size
        self.result["量感分析"] = face_volume
        self.result["面部留白"] = face_whitespace
        self.result["综合曲直"] = "曲" if total_curve_score > 0 else "直" if total_curve_score < 0 else "适中"
        self.result["推荐风格"] = style
        
        return self.result