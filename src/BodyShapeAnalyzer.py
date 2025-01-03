import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class PoseAnalyzer:
    def __init__(self, image):
        self.image = image
        # self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False)
        self.landmarks = None
        self.result = {}

    def process_image(self):
        results = self.pose.process(self.image_rgb)
        if results.pose_landmarks:
            self.landmarks = results.pose_landmarks.landmark
            return results
        else:
            raise ValueError("No pose landmarks detected.")

    def get_point(self, landmark):
        return landmark.x, landmark.y

    def calculate_distances(self):
        point4 = self.get_point(self.landmarks[4])
        point10 = self.get_point(self.landmarks[10])
        dist_4_10 = np.linalg.norm(np.array(point4) - np.array(point10))

        point1 = self.get_point(self.landmarks[1])
        point9 = self.get_point(self.landmarks[9])
        dist_1_9 = np.linalg.norm(np.array(point1) - np.array(point9))

        head_height = max(dist_4_10, dist_1_9) * 2

        left_shoulder = self.get_point(self.landmarks[11])
        right_shoulder = self.get_point(self.landmarks[12])
        shoulder_width = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))

        shoulder_ratio = shoulder_width / head_height
        return head_height, shoulder_width, shoulder_ratio

    def calculate_body_proportions(self, head_height):
        left_shoulder = self.get_point(self.landmarks[11])
        right_shoulder = self.get_point(self.landmarks[12])
        mid_shoulder = ((left_shoulder[0] + right_shoulder[0]) / 2, (left_shoulder[1] + right_shoulder[1]) / 2)

        left_hip = self.get_point(self.landmarks[23])
        right_hip = self.get_point(self.landmarks[24])
        mid_hip = ((left_hip[0] + right_hip[0]) / 2, (left_hip[1] + right_hip[1]) / 2)

        upper_body_length = np.linalg.norm(np.array(mid_shoulder) - np.array(mid_hip))

        left_ankle = self.get_point(self.landmarks[31])
        right_ankle = self.get_point(self.landmarks[32])
        mid_ankle = ((left_ankle[0] + right_ankle[0]) / 2, (left_ankle[1] + right_ankle[1]) / 2)

        lower_body_length = np.linalg.norm(np.array(mid_hip) - np.array(mid_ankle))

        upper_body_ratio = upper_body_length / head_height
        lower_body_ratio = lower_body_length / head_height
        body_ratio = lower_body_ratio / upper_body_ratio
        return upper_body_ratio, lower_body_ratio, body_ratio

    def analyze_proportions(self, body_ratio):
        if body_ratio > 1:
            if 1 <= body_ratio <= 1.25:
                return "五五身"
            elif 1.26 <= body_ratio <= 1.55:
                return "四六身（显高）"
            elif body_ratio > 1.56:
                return "三七身（黄金比例）"
        else:
            return "六四分（显矮）"

    def analyze(self):
        results = self.process_image()
        annotated_image = self.image.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )

        head_height, shoulder_width, shoulder_ratio = self.calculate_distances()
        if shoulder_ratio < 1.5:
            shoulder_type = "肩偏窄"
        elif shoulder_ratio > 1.7:
            shoulder_type = "肩偏宽"
        else:
            shoulder_type = "肩正常"

        upper_body_ratio, lower_body_ratio, body_ratio = self.calculate_body_proportions(head_height)
        proportion_type = self.analyze_proportions(body_ratio)

        # 输出结果
        # print(f"头肩比（标准化肩宽）：{shoulder_ratio:.2f}")
        # print(f"头肩比判断：{shoulder_type}")
        # print(f"上半身长度（标准化）：{upper_body_ratio:.2f}")
        # print(f"下半身长度（标准化）：{lower_body_ratio:.2f}")
        # print(f"上下半身比例：{body_ratio:.2f}")
        # print(f"比例判断：{proportion_type}")
        self.result['头肩比'] = f"{shoulder_ratio:.2f}"
        self.result['头肩比判断'] = f"{shoulder_type}"
        self.result['上半身长度'] = f"{upper_body_ratio:.2f}"
        self.result['下半身长度'] = f"{lower_body_ratio:.2f}"
        self.result['上下半身比例'] = f"{body_ratio:.2f}"
        self.result['身材比例判断'] = f"{proportion_type}"
        
        return self.result
        
        
    def close(self):
        self.pose.close()

#图像分割，提取轮廓
class ImageSegmentationProcessor:
    def __init__(self, image, model_path = 'selfie_segmenter.tflite'):
        self.model_path = model_path
        self.image = image
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.segmenter = None
        self.output_image = None  # 新增类变量存储输出图像  三通道二值图
        self.output_image_Grey = None # 单通道二值灰度
        self.contours = None

        # Constants
        self.BG_COLOR = (192, 192, 192)  # gray 定义背景的颜色
        self.DEFAULT_MASK_COLOR = np.array([0, 0, 255])  # 默认蓝色遮罩
        self.ALTERNATIVE_MASK_COLOR = np.array([255, 255, 0])  # 替代的黄色遮罩 如果是蓝色背景下
        self.DESIRED_HEIGHT = 480   #输出图像的尺寸
        self.DESIRED_WIDTH = 480

    def initialize_segmenter(self):  #初始化图像分割模型
        base_options = python.BaseOptions(model_asset_path=self.model_path)  #选择预训练模型
        options = vision.ImageSegmenterOptions(base_options=base_options, output_category_mask=True) #生成类别遮罩
        self.segmenter = vision.ImageSegmenter.create_from_options(options)  # 创建分割器

    def check_background_color(self, image, mask):  
        """
        检查背景区域的主要颜色
        注意要用RGB图像 我在类开始定义了BGR2RGB
        """
        # 获取背景区域
        background_mask = ~mask   #遮罩取反
        background_pixels = image[background_mask]   # 获取背景像素  background_mask是一个布尔数组 判断为True则属于背景

        if len(background_pixels) == 0:  #这里raise 一个错误
            return False

        # 计算背景区域的平均颜色
        avg_color = np.mean(background_pixels, axis=0)

        # 判断蓝色是否为主要颜色
        is_bluish = (avg_color[2] > avg_color[0] * 1.2) and (avg_color[2] > avg_color[1] * 1.2)

        """
        在 OpenCV 中，图像的颜色通道顺序是 BGR(蓝、绿、红), 
        avg_color[2] 代表红色通道的平均值, avg_color[1] 代表绿色通道的平均值, avg_color[0] 代表蓝色通道的平均值,
        1.2说明蓝色像素比其他两个颜色都高出20%，也可以指定其他值。这样做适合简单背景的人物照片，如果后续有复杂任务，欢迎修改。
        """

        return is_bluish

    def resize_and_show(self, image):
        h, w = image.shape[:2]
        if h < w:
            img = cv2.resize(image, (self.DESIRED_WIDTH, math.floor(h / (w / self.DESIRED_WIDTH))))
        else:
            img = cv2.resize(image, (math.floor(w / (h / self.DESIRED_HEIGHT)), self.DESIRED_HEIGHT))
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

        
    def print_contours(self, contours):
        for contour_index, contour in enumerate(contours):
            print(f"Contour {contour_index}:")
            for point in contour:
                x, y = point[0]
                print(f"({x}, {y})")

    def process_image(self):
        # 如果分割器未初始化，则初始化
        if not self.segmenter: 
            self.initialize_segmenter()

        # 创建MediaPipe图像
        mp_image = mp.Image.create_from_file(self.image_path)

        #获取分割的结果并创建一个透明的遮罩
        segmentation_result = self.segmenter.segment(mp_image)
        category_mask = segmentation_result.category_mask      # 获取类别遮罩 是一个numpy数组，0代表人物 1代表背景

        category_mask_array = category_mask.numpy_view()    # 将 category_mask 转换为 NumPy 数组   uint8 类型
        print(category_mask_array.dtype)
        

       
        overlay = self.image_rgb.copy()  # 创建透明覆盖层

        
        alpha = 1 #透明度

        
        condition = category_mask_array <= 0.2  # 创建遮罩条件  置信度小于等于0.2 就是False代表人 

        
        _, binary_mask = cv2.threshold(category_mask_array, 127, 255, cv2.THRESH_BINARY)
        binary_mask = binary_mask.astype(np.uint8) #二值化数组 0-255 其中0代表人 255代表背景
        binary_mask = cv2.bitwise_not(binary_mask) #取反 方便findContours使用
        
        plt.imshow(binary_mask, cmap='gray')  # 使用灰度色彩映射
        plt.axis('off')  # 关闭坐标轴
        plt.title("Binary Mask Visualization")  # 设置标题
        plt.show()  # 显示图像

        self.contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #提取所有轮廓而不是外轮廓
        # self.print_contours(self.contours) 打印轮廓坐标
        
        # 检查背景颜色
        is_blue_background = self.check_background_color(self.image_rgb, ~condition)

        # 根据背景颜色选择遮罩颜色
        MASK_COLOR = self.ALTERNATIVE_MASK_COLOR if is_blue_background else self.DEFAULT_MASK_COLOR

        # 创建彩色遮罩
        colored_mask = np.zeros_like(self.image_rgb)  #创建一个全0的图层 但是大小和image一致
        colored_mask[condition] = MASK_COLOR

        # 使用cv2.addWeighted合并原图和遮罩
        self.output_image = cv2.addWeighted(self.image_rgb, 1, colored_mask, alpha, 0)  # 保存到类变量
        
        # 在 output_image 上绘制轮廓
        cv2.drawContours(self.output_image, self.contours, -1, (255, 255, 255), 2)  # 使用绿色绘制轮廓
        
        # 确保输出图像不为空
        if self.output_image is None:
            raise ValueError("Segmentation processing failed. No output image generated.")
            
        
        # 打印使用了哪种颜色的遮罩
        color_used = "黄色" if is_blue_background else "蓝色"
        print(f'使用了{color_used}遮罩，因为背景{"是" if is_blue_background else "不是"}蓝色的')

        # 显示结果
        self.resize_and_show(self.output_image)

        # 返回处理后的图像
        return self.output_image
    
class PoseSegmentationVisualizer:
    def __init__(self, pose_image_path):
        self.pose_analyzer = PoseAnalyzer(pose_image_path)
        self.segmentation_processor = ImageSegmentationProcessor(pose_image_path)
        self.intersection_points = []  # 存储交点的列表
        self.widths = {} #储存宽度的字典
        self.bodytype = ""  # 储存身体形状的变量
        self.result = {}
        
    def visualize_specific_landmarks(self, image, landmarks, indices):
        """
        在图像上绘制指定的 MediaPipe 关键点。
        :param image: 要绘制的图像
        :param landmarks: MediaPipe 关键点列表
        :param indices: 要绘制的关键点索引列表
        """
        for index in indices:
            if index < len(landmarks):
                landmark = landmarks[index]
                x = int(landmark.x * image.shape[1])
                y = int(landmark.y * image.shape[0])
                cv2.circle(image, (x, y), 5, (255, 255, 255), -1)  # 红色圆圈表示指定的关键点
                cv2.putText(image, str(index), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    def visualize_combined(self):
        # 获取遮罩处理后的图像
        segmented_image = self.segmentation_processor.process_image()

        # 处理姿态分析
        self.pose_analyzer.process_image()
        landmarks = self.pose_analyzer.landmarks

        # 在分割图像上绘制姿态关键点
        for idx, landmark in enumerate(landmarks):
            x = int(landmark.x * segmented_image.shape[1])
            y = int(landmark.y * segmented_image.shape[0])
            cv2.circle(segmented_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(segmented_image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        # 显示结果
        plt.figure(figsize=(12, 12))
        plt.imshow(segmented_image)
        plt.axis('off')
        plt.title("Pose and Segmentation Visualization")
        plt.show()

    def calculate_Body_midpoints(self, landmarks):
        # 计算肩部和臀部的中点
        shoulder_mid = self.calculate_midpoint(landmarks[11], landmarks[12])
        hip_mid = self.calculate_midpoint(landmarks[23], landmarks[24])

        # 计算胸部和腰部的中点
        chest_mid, waist_mid = self.calculate_chest_waist_midpoints(shoulder_mid, hip_mid)

        return shoulder_mid, chest_mid, waist_mid, hip_mid

    def calculate_midpoint(self, point1, point2):
        # 计算两个点的中点
        x = (point1.x + point2.x) / 2
        y = (point1.y + point2.y) / 2
        return {'x': x, 'y': y}

        
    def calculate_distance(self, point1, point2):
        # 计算两个关键点之间的欧几里得距离
        return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

    def calculate_chest_waist_midpoints(self, shoulder_mid, hip_mid):
        # 计算胸部和腰部的中点
        chest_mid = {
            'x': (2 * shoulder_mid['x'] + hip_mid['x']) / 3,
            'y': (2 * shoulder_mid['y'] + hip_mid['y']) / 3
        }
        waist_mid = {
            'x': (shoulder_mid['x'] + 2* hip_mid['x']) / 3,
            'y': (shoulder_mid['y'] + 2* hip_mid['y']) / 3
        }
        return chest_mid, waist_mid

    def find_nearest_intersections(self, keypoint, contours):
        """
        用于找与指定关键点左右两边最近的轮廓点
        把轮廓点成对的形式循环遍历，找到包含关键点的y坐标范围的点对然后计算最近距离
        """
        x = keypoint['x']
        y = keypoint['y']
    
        # 查找可能的交点
        left_intersections = []
        right_intersections = []
        for contour in contours:
            for i in range(len(contour) - 1):
                pt1 = contour[i][0]
                pt2 = contour[i + 1][0]
    
                # 将轮廓点转换为归一化坐标
                pt1_normalized = (pt1[0] / self.segmentation_processor.output_image.shape[1], 
                                  pt1[1] / self.segmentation_processor.output_image.shape[0])
                pt2_normalized = (pt2[0] / self.segmentation_processor.output_image.shape[1], 
                                  pt2[1] / self.segmentation_processor.output_image.shape[0])
    
                # 检查关键点的 y 是否在当前线段的 y 范围内
                if min(pt1_normalized[1], pt2_normalized[1]) <= y <= max(pt1_normalized[1], pt2_normalized[1]):
                    # 根据 x 坐标将交点分为左侧和右侧
                    if pt1_normalized[0] < x:
                        left_intersections.append(pt1_normalized)
                    if pt2_normalized[0] < x:
                        left_intersections.append(pt2_normalized)
                    if pt1_normalized[0] > x:
                        right_intersections.append(pt1_normalized)
                    if pt2_normalized[0] > x:
                        right_intersections.append(pt2_normalized)
    
        # 找到最近的左侧和右侧交点，考虑 y 坐标的接近程度
        left_intersections.sort(key=lambda p: (abs(p[0] - x), abs(p[1] - y)))
        right_intersections.sort(key=lambda p: (abs(p[0] - x), abs(p[1] - y)))
    
        nearest_left = left_intersections[0] if left_intersections else None
        nearest_right = right_intersections[0] if right_intersections else None
    
        # 计算左右交点之间的距离
        if nearest_left and nearest_right:
            distance_between_points = np.linalg.norm(np.array(nearest_left) - np.array(nearest_right))
        else:
            distance_between_points = None
    
        return [nearest_left, nearest_right], distance_between_points
    
    def calculate_intersections(self, point1, point2, contours, image, landmarks=None):
        """
        获取两个关键点中间的轮廓点并计算距离
        :param point1: 第一个关键点的索引或字典
        :param point2: 第二个关键点的索引或字典
        :param contours: 图像的轮廓
        :param image: 用于可视化的图像
        :param landmarks: MediaPipe 关键点列表（可选）
        """
        # 如果 point1 和 point2 是index，则从 landmarks 中获取坐标放入字典，整体用字典来处理坐标
        if isinstance(point1, int) and isinstance(point2, int) and landmarks is not None:
            keypoint1 = {'x': landmarks[point1].x, 'y': landmarks[point1].y}
            keypoint2 = {'x': landmarks[point2].x, 'y': landmarks[point2].y}
        else:
            keypoint1 = point1
            keypoint2 = point2
    
        # 获取关键点的交点
        intersections_1, _ = self.find_nearest_intersections(keypoint1, contours)
        intersections_2, _ = self.find_nearest_intersections(keypoint2, contours)
    
        # 合并交点并筛选出横坐标在 keypoint1 和 keypoint2 之间的交点
        all_intersections = intersections_1 + intersections_2
        x_min = min(keypoint1['x'], keypoint2['x'])
        x_max = max(keypoint1['x'], keypoint2['x'])
    
        filtered_intersections = [pt for pt in all_intersections if x_min <= pt[0] <= x_max]
    
        # 计算距离
        if len(filtered_intersections) == 2:
            pt1, pt2 = filtered_intersections[:2]
            distance = self.calculate_distance(
                type('Point', (object,), {'x': pt1[0], 'y': pt1[1]}),
                type('Point', (object,), {'x': pt2[0], 'y': pt2[1]})
            )
        elif len(filtered_intersections) == 1:
            distance = 0
            
        elif len(filtered_intersections) == 0:
            distance = self.calculate_distance(
            type('Point', (object,), {'x': keypoint1['x'], 'y': keypoint1['y']}),
            type('Point', (object,), {'x': keypoint2['x'], 'y': keypoint2['y']})
        )
        else:
            distance = "距离获取失败，请检测环境和光线重新分析"
    
        # 可视化筛选后的交点
        self.visualize_intersections(keypoint1, filtered_intersections, image)
        self.visualize_intersections(keypoint2, filtered_intersections, image)
    
        print(f"关键点 {point1} 和 {point2} 之间的轮廓点距离为", distance)
        return filtered_intersections, distance

    
    def visualize_intersections(self, keypoint, intersections, image):
        # 在图像上绘制关键点、交点和连线
        x = int(keypoint['x'] * image.shape[1])
        y = int(keypoint['y'] * image.shape[0])

        # 绘制关键点
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)  #Red -- mid_point

        # 绘制交点连线
        if len(intersections) == 2:
            pt1 = (int(intersections[0][0] * image.shape[1]),
                   int(intersections[0][1] * image.shape[0]))
            pt2 = (int(intersections[1][0] * image.shape[1]),
                   int(intersections[1][1] * image.shape[0]))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)

        # 绘制交点
        for intersection in intersections:
            ix = int(intersection[0] * image.shape[1])
            iy = int(intersection[1] * image.shape[0])
            cv2.circle(image, (ix, iy), 5, (255, 255, 0), -1)


        
    def determine_body_shape(self, shoulder_width, waist_width, hip_width):
        # 判断身体形状
        
        shoulder_ratio = 1  # 以肩围为标准
        waist_ratio = waist_width / shoulder_width
        hip_ratio = hip_width / shoulder_width
        
        BWH_ratio = f"1:{waist_ratio }:{hip_ratio}"
        self.result["身材比例(肩：腰：臀)"] = BWH_ratio

        # 如果腰部和臀部比例与肩部相比小于0.2，说明约等
        if abs(shoulder_ratio - waist_ratio) < 0.1 and abs(shoulder_ratio - hip_ratio) < 0.1:
            return "H型"
            
        # 如果臀部比例与肩部相比小于0.2，同时肩部和臀部大于腰部（按照上面逻辑是大于0.2的）
        elif abs(shoulder_ratio - hip_ratio) < 0.1 and waist_ratio < shoulder_ratio and waist_ratio < hip_ratio:
            return "X型"
            
       # 如果臀部比例与肩部相比小于0.2，同时肩部和臀部小于腰部     
        elif waist_ratio > shoulder_ratio and waist_ratio > hip_ratio and abs(shoulder_ratio - hip_ratio) < 0.1:
            return "O型"

        
        elif shoulder_ratio < hip_ratio:
            return "A型"

        
        elif shoulder_ratio > hip_ratio:
            return "T型"

        
        return "Unknown"

    def analyze_leg_shape(self, distance_lap, distance_ankles, distance_calf):
        # 分析腿型 图像分辨率：低分辨率图像可能导致坐标计算不精确
        if distance_lap <= 0.02 and distance_ankles <= 0.02 and  distance_calf <= 0.03: #三个距离都小  
            return "正常腿型"
            
        elif distance_lap > 0.02 and distance_ankles <= 0.02 and distance_lap > distance_calf:   #膝盖距离最大 脚踝距离小 
            return "O型"
            
        elif distance_lap > distance_calf > distance_ankles:   #膝盖距离最大 脚踝距离小 
            return "O型倾向"
            
        elif distance_lap <= 0.02 and distance_ankles > 0.02: #膝盖距离小，脚踝距离大
            return "X型"
            
        elif distance_ankles > distance_calf > distance_lap: #膝盖距离小，脚踝距离大
            return "X型倾向"      
            
        elif distance_lap <= 0.02 and distance_ankles <= 0.02 and distance_calf > 0.02:
            return "XO型"
            
        elif distance_calf > distance_lap > distance_ankles:
            return "XO型倾向"
            
        else:
            return "未知腿型，请检查光线和环境，调整站姿重新获取"

    def process_and_visualize(self):
        # 获取遮罩处理后的图像
        segmented_image = self.segmentation_processor.process_image()

        # 处理姿态分析
        self.pose_analyzer.process_image()
        landmarks = self.pose_analyzer.landmarks
        
        # 计算肩宽为关键点11和12之间的距离并可视化
        shoulder_width = self.calculate_distance(landmarks[11], landmarks[12])
        self.widths["shoulder_width"] = 1.1 * shoulder_width   #女生的肩部是11和12距离的1.1倍，如果健身导致肩部很宽 需要换为之前的算法
        
        pt1 = (int(landmarks[11].x * segmented_image.shape[1]), int(landmarks[11].y * segmented_image.shape[0]))
        pt2 = (int(landmarks[12].x * segmented_image.shape[1]), int(landmarks[12].y * segmented_image.shape[0]))
        cv2.circle(segmented_image, pt1, 5, (255, 255, 255), -1)
        cv2.circle(segmented_image, pt2, 5, (255, 255, 255), -1)
        cv2.line(segmented_image, pt1, pt2, (0, 255, 0), 2)

        
        # 计算中点
        shoulder_mid, chest_mid, waist_mid, hip_mid = self.calculate_Body_midpoints(landmarks)
        
        # 打印中点坐标
        print(f"Shoulder Mid: {shoulder_mid}")
        print(f"Chest Mid: {chest_mid}")
        print(f"Waist Mid: {waist_mid}")
        print(f"Hip Mid: {hip_mid}")

        # 查找轮廓 
        contours = self.segmentation_processor.contours

        # 定义关键点名称和对应的中点
        keypoints = {
            "chest_width": chest_mid,
            "waist_width": waist_mid,
            "hip_width": hip_mid
        }

        # 计算并可视化交点
        for key, keypoint in keypoints.items():
            intersections, distances = self.find_nearest_intersections(keypoint, contours)
            # print(f"Keypoint: {keypoint}, Intersections: {intersections}, Distances: {distances}") 
            self.visualize_intersections(keypoint, intersections, segmented_image)
            
            
            self.widths[key] = distances  #储存腰围和 臀围、胸围
        
        # 使用宽度信息判断身体形状
        self.bodytype = self.determine_body_shape(
            self.widths.get("shoulder_width", 0),
            self.widths.get("waist_width", 0),
            self.widths.get("hip_width", 0)
        )
        self.result["身材类型"] = self.bodytype

        # 可视化指定的关键点
        specific_landmarks = [23, 24,25,26,27,28,29]  # 示例：绘制关键点 0, 11, 12
        self.visualize_specific_landmarks(segmented_image, landmarks, specific_landmarks)

        # 计算膝盖和脚踝的交点和距离
        filtered_intersections_Lap, distance_lap = self.calculate_intersections(25, 26, contours, segmented_image, landmarks)
        filtered_intersections_ankles, distance_ankles = self.calculate_intersections(27, 28, contours, segmented_image, landmarks)
    
        # 计算小腿中间点
        right_leg_calf = self.calculate_midpoint(landmarks[26], landmarks[28]) #字典形式{"x":x,"y":y}
        left_leg_calf = self.calculate_midpoint(landmarks[25], landmarks[27])
        
        # 计算小腿交点距离
        filtered_intersections_calf, distance_calf = self.calculate_intersections(right_leg_calf, left_leg_calf, contours, segmented_image, landmarks)

        # 分析腿型
        leg_shape = self.analyze_leg_shape(distance_lap, distance_ankles, distance_calf)
        self.result["腿型"] = leg_shape
        print("Leg Shape:", leg_shape)
    
        # 显示结果
        plt.figure(figsize=(12, 12))
        plt.imshow(self.segmentation_processor.output_image)
        plt.axis('off')
        plt.title("Pose and Body type Analysis")
        plt.show()

        return self.result