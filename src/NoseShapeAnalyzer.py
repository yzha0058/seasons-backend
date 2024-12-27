import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import math
import numpy as np

class NoseAnalyzer:
    def __init__(self, image):
        self.image = image
        # self.image_path = image_path
        # self.image = cv2.imread(image_path)
        self.image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.landmarks_coords = []
        self.image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 添加灰度图转换
        
    def detect_landmarks(self):
        # 初始化 MediaPipe Face Mesh
        mp_face_landmarker = mp.solutions.face_mesh
        face_landmarker = mp_face_landmarker.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        # 处理图像，检测关键点
        results = face_landmarker.process(self.image_rgb)
        # 检查是否检测到面部
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * self.image.shape[1])
                    y = int(landmark.y * self.image.shape[0])
                    self.landmarks_coords.append((idx, x, y))
        print(f"检测到的关键点总数: {len(self.landmarks_coords)}")

    def calculate_distance_ratio(self):
        """计算距离比值，并判断鼻翼宽窄"""
        
        # 获取指定点的坐标
        def get_point_coords(point_idx):
            for idx, x, y in self.landmarks_coords:
                if idx == point_idx:
                    return (x, y)
            return None
    
        # 计算两点之间的距离
        def calculate_distance(p1, p2):
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
        # 获取所需点的坐标
        p45 = get_point_coords(45)
        p275 = get_point_coords(275)
        p48 = get_point_coords(48)
        p344 = get_point_coords(344)
        p155 = get_point_coords(155)
        p362 = get_point_coords(362)
    
        # 确保所有点都被找到
        if all([p45, p275, p48, p344, p155, p362]):
            # 计算相关距离
            distance_45_275 = calculate_distance(p45, p275)
            distance_48_344 = calculate_distance(p48, p344)
            distance_155_362 = calculate_distance(p155, p362)
    
            # 打印鼻梁和鼻翼间距的距离
            print(f"点45和275之间的距离: {distance_45_275:.2f}")
            print(f"鼻翼间距 (48-344): {distance_48_344:.2f}")
            print(f"眼间距 (155-362): {distance_155_362:.2f}")
    
            # 比较鼻翼间距和眼间距
            if distance_48_344 < distance_155_362:
                nose_width = "窄鼻翼"
            else:
                nose_width = "宽鼻翼"
            print(f"鼻翼宽窄判断: {nose_width}")
    
            # 计算距离比值
            ratio = distance_45_275 / distance_48_344
            print(f"距离比值 (45-275 / 64-294): {ratio:.4f}")
            return ratio, nose_width
        else:
            print("无法找到所需的关键点")
            return None, None
    
    def show_image_with_landmarks(self, selected_points=None):
        # 创建一份图像副本，防止修改原始图像
        image_copy = self.image.copy()
        
        # 如果没有指定 selected_points，则默认显示所有点
        if selected_points is None:
            selected_points = [point[0] for point in self.landmarks_coords]
            
        # 在图像上标记指定的关键点编号
        for idx, x, y in self.landmarks_coords:
            if idx in selected_points:
                cv2.putText(image_copy, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)
 
        # 使用 matplotlib 显示图像
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    def get_region_average(self, point1, point2, window_size=5):
        """获取两点之间区域的平均像素值"""
        # 获取两点坐标
        x1, y1 = None, None
        x2, y2 = None, None
        
        for idx, x, y in self.landmarks_coords:
            if idx == point1:
                x1, y1 = x, y
            elif idx == point2:
                x2, y2 = x, y
                
        if None in (x1, y1, x2, y2):
            return None
            
        # 计算中点
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        
        # 获取中点周围的区域
        half_window = window_size // 2
        region = self.image_gray[
            max(0, mid_y - half_window):min(self.image.shape[0], mid_y + half_window + 1),
            max(0, mid_x - half_window):min(self.image.shape[1], mid_x + half_window + 1)
        ]
        
        return np.mean(region)

    def analyze_nose_bridge(self, threshold=40):
        """分析山根位置（分左右脸）"""
        
        def analyze_region(pair1, pair2):
            """分析一对区域的像素差异"""
            region1 = self.get_region_average(*pair1)
            region2 = self.get_region_average(*pair2)
            if region1 is not None and region2 is not None:
                diff = abs(region1 - region2)
                return diff
            return None
    
        # 右脸分析
        print("=== 右脸分析 ===")
        right_diff1 = analyze_region((193, 189), (193, 168))
        right_diff2 = analyze_region((245, 122), (122, 6))
    
        # 打印右脸差异结果
        if right_diff1 is not None:
            print(f"右脸第一组区域差异 (193-189 vs 193-168): {right_diff1:.2f}")
        if right_diff2 is not None:
            print(f"右脸第二组区域差异 (245-122 vs 122-6): {right_diff2:.2f}")
    
        # 左脸分析
        print("=== 左脸分析 ===")
        left_diff1 = analyze_region((417, 413), (417, 168))
        left_diff2 = analyze_region((6, 351), (351, 465))
    
        # 打印左脸差异结果
        if left_diff1 is not None:
            print(f"左脸第一组区域差异 (417-413 vs 417-168): {left_diff1:.2f}")
        if left_diff2 is not None:
            print(f"左脸第二组区域差异 (6-351 vs 351-465): {left_diff2:.2f}")
    
        # 根据差异判断结果
        result = {"右脸": None, "左脸": None}
        
        # 右脸结果
        if right_diff1 is not None and right_diff1 > threshold:
            result["右脸"] = "山根偏上"
        elif right_diff2 is not None and right_diff2 > threshold:
            result["右脸"] = "山根偏下"
        else:
            result["右脸"] = "山根比较塌"
        
        # 左脸结果
        if left_diff1 is not None and left_diff1 > threshold:
            result["左脸"] = "山根偏上"
        elif left_diff2 is not None and left_diff2 > threshold:
            result["左脸"] = "山根偏下"
        else:
            result["左脸"] = "山根比较塌"
    
        print("=== 综合分析结果 ===")
        print(f"右脸: {result['右脸']}")
        print(f"左脸: {result['左脸']}")
        return result


    def visualize_nose_bridge_analysis(self):
        """可视化分析区域"""
        image_copy = self.image.copy()
        
        # 画出分析的区域
        def draw_analysis_region(p1, p2, color):
            coords1 = None
            coords2 = None
            for idx, x, y in self.landmarks_coords:
                if idx == p1:
                    coords1 = (x, y)
                elif idx == p2:
                    coords2 = (x, y)
            if coords1 and coords2:
                cv2.line(image_copy, coords1, coords2, color, 2)
                mid_x = int((coords1[0] + coords2[0]) / 2)
                mid_y = int((coords1[1] + coords2[1]) / 2)
                cv2.circle(image_copy, (mid_x, mid_y), 3, color, -1)
        
        # 画出两组分析区域
        draw_analysis_region(193, 189, (255, 0, 0))  # 蓝色
        draw_analysis_region(193, 168, (0, 255, 0))  # 绿色
        draw_analysis_region(245, 122, (255, 165, 0))  # 橙色
        draw_analysis_region(122, 6, (255, 0, 255))  # 紫色
        
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    def analyze_nose_wing_curvature(self):
        """分析鼻翼曲线弯曲程度"""
        
        def get_coords(indices):
            """获取一组关键点的坐标"""
            coords = []
            for idx, x, y in self.landmarks_coords:
                if idx in indices:
                    coords.append((x, y))
            return coords
        
        def calculate_curvature(points):
            """计算曲线的曲率"""
            if len(points) < 3:  # 至少需要3个点来拟合曲线
                return None
            
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            
            # 二次多项式
            z = np.polyfit(x, y, 2)
            
            # 曲率的近似公式：k = |2 * a|
            curvature = abs(2 * z[0])
            return curvature
    
        # 定义左右鼻翼的关键点
        right_nose_wing_indices = [235, 64, 102, 49]
        left_nose_wing_indices = [455, 294, 331, 279]
        
        # 获取左右鼻翼的坐标
        right_coords = get_coords(right_nose_wing_indices)
        left_coords = get_coords(left_nose_wing_indices)
        
        # 计算左右鼻翼的曲率
        right_curvature = calculate_curvature(right_coords)
        left_curvature = calculate_curvature(left_coords)
        
        # 分析结果
        print("=== 鼻翼曲线分析 ===")
        if right_curvature is not None:
            print(f"右鼻翼曲率: {right_curvature:.4f}")
            if right_curvature > 0.5:  # 曲率阈值，后期实验后再调节
                print("右鼻翼偏曲线感强")
            else:
                print("右鼻翼偏直线感强")
        else:
            print("无法计算右鼻翼曲率")
    
        if left_curvature is not None:
            print(f"左鼻翼曲率: {left_curvature:.4f}")
            if left_curvature > 0.5:  
                print("左鼻翼偏曲线感强")
            else:
                print("左鼻翼偏直线感强")
        else:
            print("无法计算左鼻翼曲率")
    
        return {"右鼻翼曲率": right_curvature, "左鼻翼曲率": left_curvature}

