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

        # 初始化YOLO模型（确保'yolov8n-face.pt'文件存在）
        self.yolo_model = YOLO('yolov8n-face.pt')

        # 人脸检测框初始化
        self.x_left = 0
        self.y_top = 0
        self.x_right = self.image.shape[1]
        self.y_bottom = self.image.shape[0]

    def detect_face_with_yolo(self):
        """使用YOLO检测人脸,并保存检测到的人脸框坐标"""
        results = self.yolo_model(self.image_path)
        if results[0].boxes:
            box = results[0].boxes[0]  # 假设只分析第一张人脸
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cv2.rectangle(self.image, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # 将检测框保存为类属性
            self.x_left = x1
            self.y_top = y1
            self.x_right = x2
            self.y_bottom = y2
        else:
            # 未检测到人脸则使用整张图像
            self.x_left, self.y_top, self.x_right, self.y_bottom = 0, 0, self.image.shape[1], self.image.shape[0]
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

    def calculate_five_eyes(self): #127, 356, 130, 133, 362, 359
        """计算五眼比例""" 
        x_127 = self.get_point(127)[0]
        x_356 = self.get_point(264)[0]  #356 左脸识别老是超出人脸范围
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

        # 按照之前示例,将中间的两个区当做基准为1
        # 如果您希望与之前的代码保持一致,可自行调整
        ratio_1 = round(region_1_width / region_2_width, 2)
        ratio_2 = 1
        ratio_3 = round(region_3_width / region_2_width, 2) # 之前代码中是设定中间两眼为1:1:1:...
        ratio_4 = round(region_4_width / region_2_width, 2)
        ratio_5 = round(region_5_width / region_2_width, 2)

        self.result["五眼比例"] = f"{ratio_1} : {ratio_2} : {ratio_3} : {ratio_4} : {ratio_5}"
        return [ratio_1, ratio_2, ratio_3, ratio_4, ratio_5]

    def calculate_three_ratios(self):
        """计算三庭比例"""
        y_keypoint_9 = int(self.landmarks[9].y * self.image.shape[0])
        y_keypoint_2 = int(self.landmarks[2].y * self.image.shape[0])

        # 使用 YOLO 的人脸框来定义 top 和 bottom
        y_top = self.y_top
        y_bottom = self.y_bottom

        region_1_height = y_keypoint_9 - y_top
        region_2_height = y_keypoint_2 - y_keypoint_9
        region_3_height = y_bottom - y_keypoint_2

        if region_1_height == 0:
            raise ValueError("Top region height is zero, cannot calculate ratios.")

        ratio_1 = 1
        ratio_2 = round(region_2_height / region_1_height, 2)
        ratio_3 = round(region_3_height / region_1_height, 2)

        self.result["三庭比例"] = f"{ratio_1} : {ratio_2} : {ratio_3}"
        return [ratio_1, ratio_2, ratio_3]

    def calculate_face_ratios(self):
        """计算三线比例和脸长脸宽比例（基于人脸框的顶点和中心点）"""

        # 计算额头宽度 (21, 251) （54.284）
        point_21 = self.get_point(21)
        point_251 = self.get_point(251)
        forehead_width = np.linalg.norm(point_21 - point_251)

        # 计算颧骨宽度 (234, 454)
        point_234 = self.get_point(234)
        point_454 = self.get_point(454)
        cheekbone_width = np.linalg.norm(point_234 - point_454)

        # 计算下颌宽度 (58, 288)
        point_58 = self.get_point(58)
        point_288 = self.get_point(288)
        jaw_width = np.linalg.norm(point_58 - point_288)

        # 使用人脸框中点作为参考：从y_top到关键点152的距离作为脸长
        x_center = (self.x_left + self.x_right) // 2
        point_top_center = np.array([x_center, self.y_top])
        point_152 = self.get_point(152)
        face_length = np.linalg.norm(point_top_center - point_152)

        # 计算比例
        forehead_ratio = round(forehead_width / cheekbone_width, 2)
        jaw_ratio = round(jaw_width / cheekbone_width, 2)
        face_length_ratio = round(face_length / cheekbone_width, 2)

        

        print(f"三线比例 (额头: 颧骨 : 下颌) = {forehead_ratio} : 1 : {jaw_ratio}")
        print(f"脸长和脸宽的比例: {face_length_ratio}")
        self.result["三线比例"] = f"{forehead_ratio} : 1 : {jaw_ratio}"
        self.result["脸长和脸宽的比例"] = f"{face_length_ratio}"

        # 可视化测量线条
        cv2.line(self.image_rgb, tuple(point_21), tuple(point_251), (255, 0, 0), 2)  # 额头宽度
        cv2.line(self.image_rgb, tuple(point_234), tuple(point_454), (0, 255, 0), 2)  # 颧骨宽度
        cv2.line(self.image_rgb, tuple(point_58), tuple(point_288), (0, 0, 255), 2)  # 下颌宽度
        cv2.line(self.image_rgb, tuple(point_top_center), tuple(point_152), (255, 255, 0), 2)  # 脸长

        return face_length_ratio

    def analyze_chin_shape(self):
        """分析下巴形状"""
        chin_points_indices = [58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288]
        chin_points = [self.get_point(index) for index in chin_points_indices]

        point_58 = self.get_point(58)
        point_288 = self.get_point(288)
        midpoint_58_288 = (point_58 + point_288) // 2
        point_152 = self.get_point(152)

        chin_width = np.linalg.norm(midpoint_58_288 - point_152)
        chin_length = np.linalg.norm(point_58 - point_288)
        chin_length_width_ratio = chin_length / chin_width 

        # 判断下巴形状
        if chin_length_width_ratio < 2.27:
            chin_type = "锐弧（尖形下巴）"
        elif 2.27 <= chin_length_width_ratio <= 2.8:
            chin_type = "钝弧（圆形下巴）"
        else:
            chin_type = "机器人下巴（方形下巴）"

        print(f"下巴形状: {chin_type}")
        print(f"下巴宽长比: {chin_length_width_ratio:.2f}")
        self.result["下巴形状"] = f"{chin_type}"

        # 在图像上绘制下巴连接线
        for i in range(len(chin_points) - 1):
            cv2.line(self.image_rgb, tuple(chin_points[i]), tuple(chin_points[i + 1]), (0, 255, 0), 2)

        return chin_type

    def determine_face_shape(self):
        """根据重新梳理的逻辑判断脸型"""

        # 获取所需比例和数据
        forehead_width = np.linalg.norm(self.get_point(21) - self.get_point(251))
        cheekbone_width = np.linalg.norm(self.get_point(234) - self.get_point(454))
        jaw_width = np.linalg.norm(self.get_point(58) - self.get_point(288))

        face_length_ratio = self.calculate_face_ratios()
        chin_type = self.analyze_chin_shape()
        five_eyes_ratios = self.calculate_five_eyes()
        ratio_1, _, _, _, ratio_5 = five_eyes_ratios

        face_shape = ""


        # 如果颧骨最宽,并且和额头、下颌的比例差小于0.11,下巴不是锐弧,那么就是方形脸或者方菱脸
        if (cheekbone_width > forehead_width and cheekbone_width > jaw_width):
            if (abs(cheekbone_width - forehead_width) / cheekbone_width <= 0.11) and (abs(cheekbone_width - jaw_width) / cheekbone_width <= 0.11):
                if chin_type != "锐弧（尖形下巴）":
                    if ratio_1 < 0.75 and ratio_5 < 0.75:
                        face_shape = "方菱脸"
                    elif ratio_1 < 0.75 or ratio_5 < 0.75:
                        face_shape = "方形脸,轻度菱形"
                    else:
                        face_shape = "方形脸"
                else:
                    face_shape = "甲子脸" #比例相近的情况下,下巴是尖的
            else:   
                #这里可能颧骨明显宽了
                # 菱形脸（伴随太阳穴凹陷）判断 ratio_1 < 0.7 且 ratio_5 < 0.7
                if ratio_1 < 0.75 and ratio_5 < 0.75:
                    face_shape = "菱形脸"
                # 颧骨最大但不是菱形脸,进入第一套 椭圆/圆/方判断
                
                # 椭圆脸（鹅蛋脸） 
                # 条件：额头和下颌接近,face_length_ratio在[1.4,1.6],下巴不为机器人下巴
                elif abs(forehead_width - jaw_width) < 0.10 * forehead_width and (1.36 < face_length_ratio <= 1.6) and (chin_type != "机器人下巴（方形下巴）"):
                    face_shape = "椭圆脸（鹅蛋脸）"

                #倒三角脸判断 额头和颧骨相差不多时,额头明显大于下颌 宽额头对应倒三角   下巴不为机器人下巴
                elif (forehead_width > jaw_width) and abs(forehead_width - jaw_width) > 0.10 * forehead_width and (1.36 < face_length_ratio <= 1.6):
                    if chin_type != "机器人下巴（方形下巴）":
                        face_shape = "倒三角脸"
                    else:
                        face_shape = "甲子脸"

                # 额头明显比下颌小,窄额头对应菱形脸
                elif (forehead_width < jaw_width) and abs(forehead_width - jaw_width) > 0.10 * forehead_width and (1.36 < face_length_ratio <= 1.6):
                        face_shape = "菱形脸"
                
                # 圆形脸
                # 条件：额头宽度与下颌宽度差在额头宽度10%以内、脸长宽比例<1.4、下巴为钝弧（圆下巴）
                elif (abs(forehead_width - jaw_width) < 0.15 * forehead_width) and (face_length_ratio <= 1.36) and (chin_type == "钝弧（圆形下巴）"):
                    face_shape = "圆形脸"
                    




                    
        else:
            # 颧骨不是最大,进入第二套判断：
            # 不考虑颧骨最小的情况,就只有额头大于颧骨大于下颌,或者下颌大于颧骨大于额头两种情况,外加一个长脸
            # 方形脸（第二次出现）

            #下颌大于颧骨的情况
            if face_length_ratio > 1.6:
                  face_shape = "长脸"
            
            elif jaw_width >= cheekbone_width :
                  if (chin_type == "锐弧（尖形下巴）"):
                        face_shape = "方形脸"
                
            #下颌和颧骨约等,在0.1内,下巴钝弧--方形脸    
                  elif (jaw_width - cheekbone_width)/jaw_width <= 0.1 and (chin_type == "钝弧（圆形下巴）"):
                        face_shape = "方形脸"
                
                    
                #下颌大于等于颧骨,下巴是机器人下巴,或者下颌不大于颧骨0.1,下巴是钝弧,梨形脸
                  elif (chin_type == "机器人下巴（方形下巴）") or ( (jaw_width - cheekbone_width)/jaw_width >= 0.1 and (chin_type == "钝弧（圆形下巴）")):
                        face_shape = "梨形脸"   
                      

            # 倒三角脸：额头宽 > 颧骨宽 > 下颌宽 
            elif (forehead_width >= cheekbone_width > jaw_width) :
                  if (chin_type != "机器人下巴（方形下巴）"):
                        face_shape = "倒三角脸"
                  else:
                        face_shape = "方形脸"

                
            # 梨形脸：下颌宽 ≥ 颧骨宽或者下颌宽≥额头宽（根据原逻辑,下颌相对更宽）
                


            
        # 判断脸型
        if face_shape == "":
            face_shape = "请检查光线和摄像头角度并重新拍照,或咨询工作人员"
            face_curve_straight = "未知"
        else:
            # 判断脸型的曲直
            curved_faces = ["圆形脸", "梨形脸", "椭圆脸（鹅蛋脸）"]
            straight_faces = ["倒三角脸", "方形脸", "菱形脸", "方菱脸", "长脸", "甲子脸"]
            
            if face_shape in curved_faces:
                face_curve_straight = "偏曲"
            elif face_shape in straight_faces:
                face_curve_straight = "偏直"
            else:
                face_curve_straight = "未知"

        self.result["脸型判断结果"] = f"{face_shape}"
        self.result["脸型曲直"] = face_curve_straight


    def determine_three_ratios_type(self):
        """计算三庭比例类型"""
        ratios = self.calculate_three_ratios()
        three_ratios_type = ""
        tolerance = 0.1

        # ratios: [1, ratio_2, ratio_3]
        # 这里可根据您的需要调整判定条件
        # 标准脸： 1:1:1
        if abs(ratios[0] - 1) <= tolerance and abs(ratios[1] - 1) <= tolerance and abs(ratios[2] - 1) <= tolerance:
            three_ratios_type = "黄金比例脸"
            
        # 幼态脸： 1 : 1 : 0.8    
        elif abs(ratios[0] - 1) <= tolerance and abs(ratios[1] - 1) <= tolerance and abs(ratios[2] - 0.8) <= tolerance:
            three_ratios_type = "幼态脸"
            
        # 御姐成熟脸： 1 : 1.2 : 1    
        elif abs(ratios[0] - 1) <= tolerance and abs(ratios[1] - 1.2) <= tolerance and abs(ratios[2] - 1) <= tolerance:
            three_ratios_type = "御姐成熟脸"
            
        # 高气场/明艳脸： 1 : 1.2 : 1.5    
        elif abs(ratios[0] - 1) <= tolerance and abs(ratios[1] - 1.2) <= tolerance and abs(ratios[2] - 1.5) <= tolerance:
            three_ratios_type = "高气场/明艳脸"

        #中庭很短
        elif abs(ratios[0] - 1) <= tolerance and (ratios[1] < 0.9):      
            three_ratios_type = "减龄初恋脸"

        #中庭很长
        elif abs(ratios[0] - 1) <= tolerance and (ratios[1] > 1.3): 
            three_ratios_type = "长中庭, 气质脸" 
        
        #上庭和中庭相近,下庭短
        elif abs(ratios[0] - 1) <= tolerance and abs(ratios[1] - 1) <= tolerance and (ratios[2]< 0.7): 
            three_ratios_type = "可爱娃娃脸" 
            
        #上庭和中庭相近,下庭长
        elif abs(ratios[0] - 1) <= tolerance and abs(ratios[1] - 1) <= tolerance and (ratios[2]> 1.1): 
            three_ratios_type = "英气脸" 
            

        elif abs(ratios[0] - 1) <= tolerance and abs(ratios[1] - 1.2) <= tolerance and (1.1 <ratios[2]< 1.4): 
            three_ratios_type = "优雅成熟脸" 

        else:
            three_ratios_type = "温柔气质脸" 
        
        if three_ratios_type:
            print(f"三庭比例类型: {three_ratios_type}")

        self.result["脸部风格"] = f"{three_ratios_type}"

    def analyze(self):
        """主分析流程"""
        self.detect_face_with_yolo()
        if not self.extract_landmarks_with_mediapipe():
            return

        self.calculate_five_eyes()
        # 在确定三庭比例类型前需要先计算三庭比例
        # determine_three_ratios_type中已经会调用calculate_three_ratios
        # 这里可以不必重复调用calculate_three_ratios,但为了与之前代码一致性保留一次
        self.calculate_three_ratios()  
        self.calculate_face_ratios()   # 计算脸长/脸宽比例等
        # 下巴形状和脸型在determine_face_shape()中会重复调用calculate_face_ratios和analyze_chin_shape
        # 为避免重复调用,这里可直接调用但请注意代码重复执行逻辑
        # 我们这里先调用下巴形状和脸型判断
        # 注意：determine_face_shape里已经调用了analyze_chin_shape和calculate_face_ratios,会再次执行,但不影响最终结果
        self.determine_face_shape()
        self.determine_three_ratios_type()
        print(self.result)

    def visualize_original_image(self):
        """可视化原始图像"""
        # 将原图从 BGR 转换为 RGB
        original_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 10))
        plt.imshow(original_rgb)
        plt.title("Original Image")
        plt.axis('off')
        plt.show()

    def visualize_results(self):
        """可视化结果"""
        plt.figure(figsize=(10, 10))
        plt.imshow(self.image_rgb)
        plt.axis('off')  # 隐藏坐标轴
        plt.show()


## 初始化分析器
# image_path = "images/faceshape/方脸/素人3.jpg" #eimages/SCUT-FBP-141.jpg images/body_images/517.jpg
# face_analyzer = FaceAnalyzer(image_path)
# face_analyzer.analyze()
# face_analyzer.visualize_original_image()
# face_analyzer.visualize_results()