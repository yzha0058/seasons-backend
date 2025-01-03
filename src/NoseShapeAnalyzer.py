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
        self.image_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.result = {}
        
        # 定义鼻子前端多边形点索引的固定顺序
        self.nose_polygon_indices = [4, 45, 220, 115, 48, 64, 98, 97, 2, 326, 327, 294, 278, 344, 440, 275]
        
        # 曲直权重
        self.weight_shangen = 0.45 #山根权重
        self.weight_nostril = 0.15 #鼻孔权重
        self.weight_nose_width = 0.15 #鼻翼宽度权重
        self.weight_nose_wing = 0.15 #鼻翼曲线

        # 记录各特征偏直偏曲结果，初始化为空
        self.shangen_result = None   # 山根偏直/偏曲
        self.nostril_result = None   # 鼻孔偏直/偏曲
        self.nose_width_result = None  # 宽窄鼻翼偏直/偏曲
        self.nose_wing_shape_result = None  # 鼻翼曲线偏直/偏曲

    def detect_landmarks(self):
        mp_face_landmarker = mp.solutions.face_mesh
        face_landmarker = mp_face_landmarker.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        results = face_landmarker.process(self.image_rgb)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    x = int(landmark.x * self.image.shape[1])
                    y = int(landmark.y * self.image.shape[0])
                    self.landmarks_coords.append((idx, x, y))
        print(f"检测到的关键点总数: {len(self.landmarks_coords)}")

    # 计算鼻翼宽度比例
    def calculate_distance_ratio(self):
        def get_point_coords(point_idx):
            for idx, x, y in self.landmarks_coords:
                if idx == point_idx:
                    return (x, y)
            return None
    
        def calculate_distance(p1, p2):
            return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
        p45 = get_point_coords(45)
        p275 = get_point_coords(275)
        p48 = get_point_coords(48)
        p278 = get_point_coords(278)
        p155 = get_point_coords(155)
        p362 = get_point_coords(362)
    
        if all([p45, p275, p48, p278, p155, p362]):
            distance_45_275 = calculate_distance(p45, p275)
            distance_48_278 = calculate_distance(p48, p278)
            distance_155_362 = calculate_distance(p155, p362)
    
            print(f"点45和275之间的距离: {distance_45_275:.2f}")
            print(f"鼻翼间距 (48-278): {distance_48_278:.2f}")
            print(f"眼间距 (155-362): {distance_155_362:.2f}")
    
            if distance_48_278 < distance_155_362:
                nose_width = "窄鼻翼"
                self.nose_width_result = "偏直"
            else:
                nose_width = "宽鼻翼"
                self.nose_width_result = "偏曲"
                
            self.result["鼻翼宽窄"] = f"{nose_width},{self.nose_width_result}"
            
            print(f"鼻翼宽窄判断: {nose_width}")
    
            ratio = distance_45_275 / distance_48_278
            print(f"距离比值 (45-275 / 64-294): {ratio:.4f}")
            return ratio, nose_width
        else:
            print("无法找到所需的关键点")
            return None, None
    
    def show_image_with_landmarks(self, selected_points=None):
        image_copy = self.image.copy()
        
        if selected_points is None:
            selected_points = [point[0] for point in self.landmarks_coords]
            
        for idx, x, y in self.landmarks_coords:
            if idx in selected_points:
                cv2.putText(image_copy, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
 
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    # 指定窗口大小的区域平均灰度值
    def get_region_average(self, point1, point2, window_size=5):
        x1, y1 = None, None
        x2, y2 = None, None
        
        for idx, x, y in self.landmarks_coords:
            if idx == point1:
                x1, y1 = x, y
            elif idx == point2:
                x2, y2 = x, y
                
        if None in (x1, y1, x2, y2):
            return None
            
        mid_x = int((x1 + x2) / 2)
        mid_y = int((y1 + y2) / 2)
        
        half_window = window_size // 2
        region = self.image_gray[
            max(0, mid_y - half_window):min(self.image.shape[0], mid_y + half_window + 1),
            max(0, mid_x - half_window):min(self.image.shape[1], mid_x + half_window + 1)
        ]
        
        return np.mean(region)
        
    #分析灰度差异
    def analyze_nose_bridge(self, threshold=40):
        def analyze_region(pair1, pair2):
            region1 = self.get_region_average(*pair1)
            region2 = self.get_region_average(*pair2)
            if region1 is not None and region2 is not None:
                diff = abs(region1 - region2)
                return diff
            return None
    
        print("=== 右脸分析 ===")
        right_diff1 = analyze_region((193, 189), (193, 168))
        right_diff2 = analyze_region((245, 122), (122, 6))
    
        if right_diff1 is not None:
            print(f"右脸第一组区域差异 (193-189 vs 193-168): {right_diff1:.2f}")
        if right_diff2 is not None:
            print(f"右脸第二组区域差异 (245-122 vs 122-6): {right_diff2:.2f}")
    
        print("=== 左脸分析 ===")
        left_diff1 = analyze_region((417, 413), (417, 168))
        left_diff2 = analyze_region((6, 351), (351, 465))
    
        if left_diff1 is not None:
            print(f"左脸第一组区域差异 (417-413 vs 417-168): {left_diff1:.2f}")
        if left_diff2 is not None:
            print(f"左脸第二组区域差异 (6-351 vs 351-465): {left_diff2:.2f}")
    
        result = {"右脸": None, "左脸": None}
        
        # 山根结果判定逻辑
        def judge_shangen_side(diff1, diff2):
            # diff1 > threshold => 偏上 => 偏直
            # diff2 > threshold => 偏下 => 偏曲
            # 否则 => 比较塌 => 偏曲
            if diff1 is not None and diff1 > threshold:
                return ["山根靠上","偏直"]
            elif diff2 is not None and diff2 > threshold:
                return ["山根靠下","偏曲"]
            else:
                # 山根比较塌 => 偏曲
                return ["山根比较塌","偏曲"]
        
        right_side = judge_shangen_side(right_diff1, right_diff2)
        left_side = judge_shangen_side(left_diff1, left_diff2)
        
        self.result["右脸山根"] = f"{right_side}"
        self.result["左脸山根"] = f"{left_side}"
        
        # 如果有一边是偏直，那么按偏直计算，否则偏曲
        if right_side[1] == "偏直" or left_side[1] == "偏直":
            self.shangen_result = "偏直"
        else:
            self.shangen_result = "偏曲"
            
        self.result["山根曲直"] =f"{self.shangen_result}"
        print("=== 综合分析结果 ===")
        print(f"右脸: {right_side[0]}, 左脸: {left_side[0]}, 山根综合: {self.shangen_result}")
        return result

    #鼻翼曲线分析
    def analyze_nose_wing_curvature(self):
        def get_coords(indices):
            coords = []
            for idx, x, y in self.landmarks_coords:
                if idx in indices:
                    coords.append((x, y))
            return coords
            
        #计算曲率 --鼻翼弧度反应曲直关系
        def calculate_curvature(points):
            if len(points) < 3:
                return None
            x = [p[0] for p in points]
            y = [p[1] for p in points]
            z = np.polyfit(x, y, 2)
            curvature = abs(2 * z[0])
            return curvature
    
        right_nose_wing_indices = [235, 64, 102, 49]
        left_nose_wing_indices = [455, 294, 331, 279]
        
        right_coords = get_coords(right_nose_wing_indices)
        left_coords = get_coords(left_nose_wing_indices)
        
        right_curvature = calculate_curvature(right_coords)
        left_curvature = calculate_curvature(left_coords)
        
        print("=== 鼻翼曲线分析 ===")
        # 对左右鼻翼都判断，只要有一个达到某种程度就可判断
        # 这里选择取平均或是根据要求对左右综合判断。
        # 需求未明确左右综合，这里简单判断每侧，若一侧偏曲则整体为偏曲，否则为偏直(如有需要可调整逻辑)
        
        def wing_judge(curv):
            if curv is None:
                return None
            if curv > 0.5:
                return "偏曲"
            else:
                return "偏直"
        
        right_judge = wing_judge(right_curvature)
        left_judge = wing_judge(left_curvature)
        
        # 如果有一侧偏曲 => 整体偏曲，否则偏直(可根据需求修改)
        if right_judge == "偏曲" or left_judge == "偏曲":
            self.nose_wing_shape_result = "偏曲"
        else:
            self.nose_wing_shape_result = "偏直"
        
        if right_curvature is not None:
            print(f"右鼻翼曲率: {right_curvature:.4f}, 判断: {right_judge}")
        else:
            print("无法计算右鼻翼曲率")
        if left_curvature is not None:
            print(f"左鼻翼曲率: {left_curvature:.4f}, 判断: {left_judge}")
        else:
            print("无法计算左鼻翼曲率")

        self.result["鼻翼曲线综合判断"] = self.nose_wing_shape_result
        self.result["右鼻翼曲率"] = f"{right_curvature}" 
        self.result["左鼻翼曲率"] = f"{left_curvature}" 
        
        print(f"鼻翼曲线综合判断: {self.nose_wing_shape_result}")
        return {"右鼻翼曲率": right_curvature, "左鼻翼曲率": left_curvature}
        
    #鼻孔比例计算方法
    def calculate_nostril_ratio(self):
        image_gray = self.image_gray
        polygon_coords = []
        for p_idx in self.nose_polygon_indices:
            found = False
            for idx, x, y in self.landmarks_coords:
                if idx == p_idx:
                    polygon_coords.append((x,y))
                    found = True
                    break
            if not found:
                return 0.0, "无法计算", None, None
        
        polygon_coords_array = np.array(polygon_coords, dtype=np.int32)
        
        mask = np.zeros(image_gray.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_coords_array], 255)
        
        black_threshold = 110
        region_pixels = image_gray[mask == 255]
        if len(region_pixels) == 0:
            return 0.0, "无法计算", polygon_coords_array, mask
        
        black_pixels_count = np.sum(region_pixels < black_threshold)
        total_pixels = len(region_pixels)
        black_ratio = black_pixels_count / total_pixels
        
        # 判断是否漏鼻孔
        if black_ratio > 0.06:
            nostril_status = "漏鼻孔鼻尖上翘"
            self.nostril_result = "偏曲"
        else:
            nostril_status = "正常鼻孔"
            self.nostril_result = "偏直"

        self.result["鼻孔曲直"] = self.nostril_result
        return black_ratio, nostril_status, polygon_coords_array, mask

    def visualize_nose_bridge_analysis(self):
        image_copy = self.image.copy()

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
        
        draw_analysis_region(193, 189, (255, 0, 0))  
        draw_analysis_region(193, 168, (0, 255, 0))  
        draw_analysis_region(245, 122, (255, 165, 0)) 
        draw_analysis_region(122, 6, (255, 0, 255))  

        # 计算鼻孔比例
        nostril_ratio, nostril_status, polygon_coords_array, mask = self.calculate_nostril_ratio()
        self.result["鼻孔比例"] =f"{nostril_ratio*100:.2f}%"
        
        # 高亮鼻孔区域
        if mask is not None:
            image_gray_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)
            black_mask = (image_gray_copy < 110) & (mask == 255)
            image_copy[black_mask] = (255, 0, 0)
        
        # 绘制鼻孔多边形
        if polygon_coords_array is not None:
            cv2.polylines(image_copy, [polygon_coords_array], isClosed=True, color=(0,0,255), thickness=2)

        # 计算综合结果
        # 特征结果: 山根(self.shangen_result), 鼻孔(self.nostril_result), 宽窄鼻翼(self.nose_width_result), 鼻翼曲线(self.nose_wing_shape_result)
        # 若有的结果还未确定，则默认偏直(0)
        def to_score(res):
            return 1 if res == "偏曲" else 0
        
        shangen_score = to_score(self.shangen_result) if self.shangen_result else 0
        nostril_score = to_score(self.nostril_result) if self.nostril_result else 0
        nose_width_score = to_score(self.nose_width_result) if self.nose_width_result else 0
        nose_wing_score = to_score(self.nose_wing_shape_result) if self.nose_wing_shape_result else 0

        total_score = (shangen_score * self.weight_shangen 
                       + nostril_score * self.weight_nostril 
                       + nose_width_score * self.weight_nose_width 
                       + nose_wing_score * self.weight_nose_wing)

        if total_score > 0.5:
            final_result = "偏曲"
        else:
            final_result = "偏直"

        self.result["鼻型综合曲直"] = final_result

        # text = f"鼻孔比例: {nostril_ratio*100:.2f}%\n{nostril_status}\n综合曲直结果: {final_result}"

        # 在图像上绘制文字
        # pil_image = Image.fromarray(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
        # draw = ImageDraw.Draw(pil_image)
        # font = ImageFont.truetype("wryh.ttf", 30)
        # y0, dy = 30, 40
        # for i, line in enumerate(text.split('\n')):
        #     y = y0 + i*dy
        #     draw.text((30, y), line, font=font, fill=(255, 0, 0))
        
        # final_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # plt.figure(figsize=(10, 10))
        # plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.show()

        # # 打印最终结果
        # print("=== 综合曲直分析 ===")
        # print(f"山根: {self.shangen_result}, 鼻孔: {self.nostril_result}, 宽窄鼻翼: {self.nose_width_result}, 鼻翼曲线: {self.nose_wing_shape_result}")
        # print(f"综合分值: {total_score:.2f}, 最终结果: {final_result}")
        # print(self.result)


