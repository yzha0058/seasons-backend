import os
import base64
import tempfile
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import io
import fitz
import subprocess
import time
from PIL import Image
import sys

# 路径映射 - 使用Noto Sans SC字体
FONT_REG_PATH = "font/NotoSansSC-Regular.ttf"
FONT_BOLD_PATH = "font/NotoSansSC-Bold.ttf"
FONT_HEAVY_PATH = "font/NotoSansSC-ExtraBold.ttf"
FONT_BLACK_PATH = "font/NotoSansSC-Black.ttf"
FONT_SEMIBOLD_PATH = "font/NotoSansSC-SemiBold.ttf"
FONT_BLACKITALIC_PATH = "font/NotoSans-BlackItalic.ttf"

# 注册字体
pdfmetrics.registerFont(TTFont("NotoSans-Regular", FONT_REG_PATH))
pdfmetrics.registerFont(TTFont("NotoSans-Bold", FONT_BOLD_PATH))
pdfmetrics.registerFont(TTFont("NotoSans-Heavy", FONT_HEAVY_PATH))
pdfmetrics.registerFont(TTFont("NotoSans-Black", FONT_BLACK_PATH))
pdfmetrics.registerFont(TTFont("NotoSans-SemiBold", FONT_SEMIBOLD_PATH))
pdfmetrics.registerFont(TTFont("NotoSans-BlackItalic", FONT_BLACKITALIC_PATH))
# 示例数据 - 这里使用简单的占位符数据
SAMPLE_FACE_INFO = {
    "Face_shape_info": {
        "三庭比例": "占位数据", 
        "三线比例": "占位数据",
        "下巴形状": "占位数据",
        "五眼比例": "占位数据",
        "脸长和脸宽的比例": "占位数据",
        "脸型判断结果": "椭圆脸",
        "脸型曲直": "占位数据",
        "脸部量感": "占位数据",
        "脸部风格": "占位数据"
    },
    "Lips_detailed_info": {
        "唇峰高度": "占位数据",
        "唇部曲直": "占位数据"
    },
    "eye_detailed_info": {
        "左眼内眼角角度": "占位数据",
        "左眼长高比例": "占位数据",
        "左眼特征": "占位数据",
        "右眼内眼角角度": "占位数据", 
        "右眼长高比例": "占位数据",
        "右眼特征": "占位数据",
        "左眼类型": "杏眼",
        "右眼类型": "杏眼"
    },
    "eye_shape": {
        "右眼曲直": "占位数据",
        "右眼类型": "占位数据",
        "左眼曲直": "占位数据",
        "左眼类型": "占位数据",
        "眼型曲直综合": "占位数据",
        "眼神": "占位数据"
    },
    "nose_detailed_info": {
        "鼻子长度": "占位数据",
        "鼻翼宽度": "占位数据",
        "鼻尖高度": "占位数据",
        "鼻型综合曲直": "偏直"
    },
    "nose_shape": {
        "鼻型": "占位数据"
    },
    "lip_shape": {
        "唇形": "叶形唇"
    }
}

SAMPLE_BODY_INFO = {
    "body_detailed_info": {
        "肩部特征": "占位数据",
        "腰部特征": "占位数据",
        "臀部特征": "占位数据"
    },
    "three_d_model_info": {
        "身材比例": "占位数据",
        "身材类型": "T型",
        "腿型": "X型",
        "三围比例": "占位数据"
    }
}

# 示例图片 - 创建一个占位图像
def create_placeholder_image(width=500, height=700, text="示例图片"):
    # 浅蓝色背景 (204, 229, 255) - 对应RGB值
    img = Image.new('RGB', (width, height), color=(204, 229, 255))
    
    # 保存到临时文件
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    img.save(temp_file.name, "PNG")
    temp_file.close()
    
    return temp_file.name

def generate_preview_pdf():
    try:
        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4) # 创建PDF画布，A4尺寸

        pdf.setTitle("PDF Layout Preview")
        
        page_width, page_height = A4 # A4尺寸: 595.2×841.8点
        x_start, y_start = 50, 800 # 页面基本起始位置(x, y)
        
        # 调整图片尺寸
        img_width, img_height = 220, 280  # 设置图片尺寸

        face_img_path = create_placeholder_image(300, 400, "面部图像")
        body_img_path = create_placeholder_image(300, 400, "身体图像")

        # === 🟦 背景框 ===
        box_x, box_y = 20, page_height - 300 # 背景框左下角坐标
        box_width, box_height = 310, 280  # 背景框宽度和高度
        radius = 10 # 圆角半径

        # 背景框颜色设置: RGB(0.85, 0.95, 0.9)
        pdf.setFillColorRGB(0.85, 0.95, 0.9)
        pdf.roundRect(box_x, box_y, box_width, box_height, radius, fill=1, stroke=0) # 绘制圆角矩形

        # === 🖋 SEASONS 标题 ===
        pdf.setFont("NotoSans-Black", 33) # 字体和大小
        pdf.setFillColorRGB(0, 0, 0) # 设置黑色文字
        title_x = box_x + 15 # 标题X坐标
        title_y = box_y + box_height - 40 # 标题Y坐标

        # 绘制"SE"部分
        pdf.drawString(title_x, title_y, "SE")

        # 计算"SE"的宽度并获取绘制"A"的起始位置
        se_width = pdf.stringWidth("SE", "NotoSans-Black", 33)
        a_x = title_x + se_width + 2  # 减少2像素使A向左移动，更靠近E

        # 直接使用斜体字体绘制"A"
        pdf.setFont("NotoSans-BlackItalic", 35)
        pdf.drawString(a_x, title_y, "A")

        # 计算"A"的宽度并获取绘制"SONS"的起始位置
        a_width = pdf.stringWidth("A", "NotoSans-BlackItalic", 35)
        sons_x = a_x + a_width

        # 绘制"SONS"部分，切换回原字体
        pdf.setFont("NotoSans-Black", 33)
        pdf.drawString(sons_x, title_y, "SONS")
        
        # === 📌 面部/体型分析副标题 ===
        pdf.setFont("NotoSans-Black", 33)
        pdf.drawString(box_x + 15, box_y + box_height - 85, "面部/体型分析")

        # === 🖼 面部图像 ===
        if face_img_path:
            pdf.drawImage(face_img_path, box_x + box_width + 20, box_y, img_width, img_height)

        # === 📋 简要信息区（六行） ===
        pdf.setFont("NotoSans-Regular", 19)
        summary_lines = [
            f"【脸型】 椭圆脸（鹅蛋脸）",
            f"【眼型】 左眼: 杏眼 / 右眼: 杏眼",
            f"【鼻型综合曲直】 偏直",
            f"【唇形】 叶形唇",
            f"【腿型】 X型",
            f"【身材类型】 T型",
        ]

        line_y = box_y + box_height - 130  # 摘要起始Y坐标
        line_spacing = 26  # 行间距
        for line in summary_lines:
            pdf.drawString(box_x + 15, line_y, line) # 调整文本左右间距
            line_y -= line_spacing # 更新Y坐标，向下绘制下一行

        # === 📏 分隔线（1）===
        divider_y = box_y - 30  # 分隔线Y坐标向下平移40点
        pdf.setStrokeColorRGB(0.6, 0.85, 0.7)  # RGB(0.6, 0.85, 0.7)
        pdf.setLineWidth(2)  # 线宽：2
        pdf.line(20, divider_y, page_width - 20, divider_y)

        # === 📊 脸型详细信息（左列）- 向下平移40点 ===
        # 标题
        pdf.setFont("NotoSans-Bold", 19)
        pdf.setFillColorRGB(0, 0, 0)  # 黑色
        pdf.drawString(x_start , divider_y - 40, "脸型详细信息")  # 向下平移
        
        # 内容项
        pdf.setFont("NotoSans-Regular", 15)
        left_column_x = x_start - 15
        data_y = divider_y - 78  # 
        
        data_items = [
            "• 三庭比例: 1 : 1.27 : 1.3",
            "• 三线比例: 0.97 : 1 : 0.88",
            "• 下巴形状: 机器人下巴 (方形下巴)",
            "• 五眼比例: 0.64 : 1 : 1.21 : 0.94 : 0.6",
            "• 脸长和脸宽的比例: 1.28"
        ]
        
        for item in data_items:
            pdf.drawString(left_column_x, data_y, item)
            data_y -= 27  # 行间距为23pt
        
        # === 📈 脸型分析结果（右列）- 向下平移40点 ===
        pdf.setFont("NotoSans-Heavy", 15)
        pdf.setFillColorRGB(0.2, 0.6, 0.8)  # 浅蓝色 RGB(0.2, 0.6, 0.8)
        
        right_column_x = page_width / 2 + 20
        result_y = divider_y - 78  # 向下平移40点
        
        result_items = [
            "• 脸型判断结果: 菱形脸",
            "• 脸型曲直: 偏直",
            "• 脸部量感: 大量感",
            "• 脸部风格: 优雅成熟脸"
        ]
        
        for item in result_items:
            pdf.drawString(right_column_x, result_y, item)
            result_y -= 30  # 行间距为23pt
        

        next_section_y = min(data_y, result_y) - 15
        # === 📏 分隔线（2）===
        divider_y = next_section_y  # 第二条分隔线Y坐标
        pdf.setStrokeColorRGB(0.6, 0.85, 0.7)  # RGB(0.6, 0.85, 0.7)
        pdf.setLineWidth(2)  # 线宽：2
        pdf.line(20, divider_y, page_width - 20, divider_y)

        # === 👁 眼部详细信息（左列）+ 眼型分析（右列）===
        eye_section_y = divider_y - 40  # 分隔线下方留出空间

        # 标题
        pdf.setFont("NotoSans-Bold", 19)
        pdf.setFillColorRGB(0, 0, 0)  # 黑色
        pdf.drawString(x_start, eye_section_y, "眼部详细信息")
        pdf.drawString(page_width / 2 + 40, eye_section_y, "眼型分析")

        # 眼部详细信息（左列）
        pdf.setFont("NotoSans-Regular", 15)

        eye_data_y = eye_section_y - 38  # 标题下方留出空间
        left_column_x = x_start - 15  # 保持与脸型详细信息一致的缩进

        # 左眼信息
        pdf.drawString(left_column_x, eye_data_y, "• 左眼内眼角角度: 47.73°")
        eye_data_y -= 27
        pdf.drawString(left_column_x, eye_data_y, "• 左眼长高比例: 2.85")
        eye_data_y -= 27
        pdf.drawString(left_column_x, eye_data_y, "• 左眼特征: 细长")

        # 增加左眼特征和右眼内眼角角度之间的间距
        eye_data_y -= 40  # 增加额外的空间

        # 右眼信息
        pdf.drawString(left_column_x, eye_data_y, "• 右眼内眼角角度: 51.34°")
        eye_data_y -= 27
        pdf.drawString(left_column_x, eye_data_y, "• 右眼长高比例: 2.92")
        eye_data_y -= 27
        pdf.drawString(left_column_x, eye_data_y, "• 右眼特征: 细长")

        # 眼型分析（右列）- 黑色常规项目
        pdf.setFont("NotoSans-Regular", 15)
        pdf.setFillColorRGB(0, 0, 0)  # 黑色

        eye_result_y = eye_section_y - 38  # 与左列起始位置相同

        # 右眼分析
        pdf.drawString(right_column_x, eye_result_y, "• 右眼曲直: 偏直")
        eye_result_y -= 27
        pdf.drawString(right_column_x, eye_result_y, "• 右眼类型: 丹凤眼")

        # 增加右眼类型和左眼曲直之间的间距
        eye_result_y -= 40  # 增加额外的空间

        # 左眼分析
        pdf.drawString(right_column_x, eye_result_y, "• 左眼曲直: 偏直") 
        eye_result_y -= 27
        pdf.drawString(right_column_x, eye_result_y, "• 左眼类型: 细长眼")

        # 增加左眼类型和眼型曲直综合之间的间距
        eye_result_y -= 40  # 增加额外的空间

        # 眼型分析 - 蓝色关键结论项
        pdf.setFont("NotoSans-Bold", 15)
        pdf.setFillColorRGB(0.2, 0.6, 0.8)  # 浅蓝色 RGB(0.2, 0.6, 0.8)

        # 综合结论
        pdf.drawString(right_column_x, eye_result_y, "• 眼型曲直综合: 偏直")
        eye_result_y -= 27
        pdf.drawString(right_column_x, eye_result_y, "• 眼神: 偏直")

        # === 📏 分隔线（3）===
        divider_y = 20  # 第三条分隔线Y坐标（页面底部）
        pdf.setStrokeColorRGB(0.6, 0.85, 0.7)  # RGB(0.6, 0.85, 0.7)
        pdf.setLineWidth(2)  # 线宽：2
        pdf.line(20, divider_y, page_width - 20, divider_y)
        
        # === 第二页内容 ===
        pdf.showPage()
        y_start = 780
        
        # === 鼻部详细信息和鼻型分析 ===
        pdf.setFont("NotoSans-Bold", 19)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.drawString(x_start, y_start, "鼻部详细信息")
        pdf.drawString(page_width / 2 + 40, y_start, "鼻型分析")
        y_start -= 38
        
        # 鼻部详细信息
        pdf.setFont("NotoSans-Regular", 15)
        left_column_x = x_start - 15
        data_y = y_start
        
        # 鼻部详细信息项
        nose_data = [
            "• 右脸山根: ['山根靠上', '偏直']",
            "• 右鼻翼曲率: 0.50",
            "• 左脸山根: ['山根靠上', '偏直']",
            "• 左鼻翼曲率: 0.50",
            "• 鼻孔比例: 49.29%"
        ]
        
        for item in nose_data:
            pdf.drawString(left_column_x, data_y, item)
            data_y -= 27
        
        # 鼻型分析
        pdf.setFont("NotoSans-Regular", 15)
        pdf.setFillColorRGB(0, 0, 0)
        nose_result_y = y_start
        
        # 鼻型分析项
        nose_results = [
            "• 鼻孔曲直: 偏曲",
            "• 山根曲直: 偏直",
            "• 鼻翼曲线综合判断: 偏曲",
            "• 鼻翼宽窄: 宽鼻翼,偏曲"
        ]
        
        for item in nose_results:
            pdf.drawString(right_column_x, nose_result_y, item)
            nose_result_y -= 30
        
        # 浅蓝色关键结论
        pdf.setFont("NotoSans-Bold", 15)
        pdf.setFillColorRGB(0.2, 0.6, 0.8)
        pdf.drawString(right_column_x, nose_result_y, "• 鼻型综合曲直: 偏直")
        nose_result_y -= 30
        
        # 更新下一部分的位置
        next_section_y = min(data_y, nose_result_y) - 15
        
        # === 📏 分隔线（4）===
        divider_y = next_section_y
        pdf.setStrokeColorRGB(0.6, 0.85, 0.7)
        pdf.setLineWidth(2)
        pdf.line(20, divider_y, page_width - 20, divider_y)
        
        # === 唇部详细信息和唇型分析 ===
        # 标题
        pdf.setFont("NotoSans-Bold", 19)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.drawString(x_start, divider_y - 40, "唇部详细信息")
        pdf.drawString(page_width / 2 + 40, divider_y - 40, "唇型分析")
        
        # 唇部详细信息
        pdf.setFont("NotoSans-Regular", 15)
        data_y = divider_y - 78
        
        # 唇部详细信息项
        lip_data = [
            "• 上下唇比例: 1.33",
            "• 嘴角: 嘴角平坦, 右嘴角倾斜度1.73°",
            "• 左嘴角倾斜度9.58°",
            "• 唇部数据: 宽唇, 偏薄, 无明显唇峰",
            "• 上唇: 偏厚偏曲",
            "• 下唇: 偏薄偏直"
        ]
        
        for item in lip_data:
            pdf.drawString(left_column_x, data_y, item)
            data_y -= 27
        
        # 唇型分析
        pdf.setFont("NotoSans-Regular", 15)
        pdf.setFillColorRGB(0, 0, 0)
        lip_result_y = divider_y - 78
        
        # 唇型分析项
        lip_results = [
            "• 唇部数据: 宽唇, 偏薄, 无明显唇峰",
            "• 曲直结果: 偏曲"
        ]
        
        for item in lip_results:
            pdf.drawString(right_column_x, lip_result_y, item)
            lip_result_y -= 30
        
        # 浅蓝色关键结论
        pdf.setFont("NotoSans-Bold", 15)
        pdf.setFillColorRGB(0.2, 0.6, 0.8)
        pdf.drawString(right_column_x, lip_result_y, "• 唇形: 薄唇")
        lip_result_y -= 30
        
        # 更新下一部分的位置
        next_section_y = min(data_y, lip_result_y) - 15
        
        # === 📏 分隔线（5）===
        divider_y = next_section_y
        pdf.setStrokeColorRGB(0.6, 0.85, 0.7)
        pdf.setLineWidth(2)
        pdf.line(20, divider_y, page_width - 20, divider_y)
        
        # === 体型分析和三维分析 ===

        # 体型分析标题
        pdf.setFont("NotoSans-Bold", 19)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.drawString(page_width / 2 + 40, divider_y - 40, "体型分析")

        # 体型分析内容
        pdf.setFont("NotoSans-Regular", 15)
        data_y = divider_y - 78

        # 体型分析项
        body_data = [
            "• 上下半身比例: 1.49",
            "• 头肩比: 1.68",
            "• 头肩比判断: 肩正常"
        ]

        for item in body_data:
            pdf.drawString(right_column_x, data_y, item)
            data_y -= 27

        # 添加蓝色身材结论
        pdf.setFont("NotoSans-Bold", 15)
        pdf.setFillColorRGB(0.2, 0.6, 0.8)
        pdf.drawString(right_column_x, data_y, "• 身材比例判断: 五五身")
        data_y -= 50  # 增加额外间距，分隔两部分内容

        # 三维分析标题
        pdf.setFont("NotoSans-Bold", 19)
        pdf.setFillColorRGB(0, 0, 0)
        pdf.drawString(page_width / 2 + 40, data_y, "三维分析")
        data_y -= 38

        # 三维分析内容
        pdf.setFont("NotoSans-Regular", 15)
        pdf.setFillColorRGB(0, 0, 0)

        # 三维分析项
        three_d_data = [
            "• 三围比例: 1:1.35:1.2"
        ]

        for item in three_d_data:
            pdf.drawString(right_column_x, data_y, item)
            data_y -= 27

        # 添加蓝色关键结论
        pdf.setFont("NotoSans-Bold", 15)
        pdf.setFillColorRGB(0.2, 0.6, 0.8)
        pdf.drawString(right_column_x, data_y, "• 腿型: X型倾向")
        data_y -= 27
        pdf.drawString(right_column_x, data_y, "• 身材类型: A型")

        # 绘制身体图像在左侧
        if body_img_path:
            # 缩小图片尺寸
            img_width, img_height = 220, 280  # 调整为更小的尺寸
            # 调整Y坐标，确保与底部保持距离
            img_x = 20
            img_y = 20  # 从页面底部向上留出更多空间，避免与底部接触
            pdf.drawImage(body_img_path, img_x, img_y, img_width, img_height)
        
        # 保存PDF
        pdf.save()
        buffer.seek(0)
        
        # 使用时间戳或临时文件名以避免文件访问冲突
        timestamp = int(time.time())
        output_path = f"preview_layout_{timestamp}.pdf"
        
        try:
            with open(output_path, "wb") as f:
                f.write(buffer.getvalue())
            return output_path  # 确保返回路径
        except PermissionError:
            # 如果出现权限错误，尝试使用临时目录
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"preview_layout_{timestamp}.pdf")
            with open(output_path, "wb") as f:
                f.write(buffer.getvalue())
            return output_path  # 确保返回路径
        
    except Exception as e:
        print(f"生成PDF时出错: {e}")
        # 创建一个空白PDF作为应急方案
        try:
            empty_buffer = io.BytesIO()
            empty_pdf = canvas.Canvas(empty_buffer, pagesize=A4)
            empty_pdf.setFont("NotoSans-Regular", 14)
            empty_pdf.drawString(50, 750, "生成PDF时出错，请检查控制台信息")
            empty_pdf.save()
            empty_buffer.seek(0)
            
            error_path = f"error_preview_{int(time.time())}.pdf"
            with open(error_path, "wb") as f:
                f.write(empty_buffer.getvalue())
            return error_path
        except:
            print("创建错误提示PDF也失败了")
            return None

def auto_open_pdf(pdf_path):
    """根据操作系统自动打开PDF文件"""
    if pdf_path is None:
        print("没有可打开的PDF文件")
        return
        
    if sys.platform.startswith('darwin'):  # macOS
        subprocess.run(['open', pdf_path], check=True)
    elif sys.platform.startswith('win'):   # Windows
        os.startfile(pdf_path)
    else:  # Linux或其他
        try:
            subprocess.run(['xdg-open', pdf_path], check=True)
        except:
            print(f"PDF已生成，请手动打开：{pdf_path}")

def watch_for_changes(target_file, interval=1.0):
    """监视目标文件的更改并在更改时重新生成PDF"""
    last_modified = os.path.getmtime(target_file)
    
    while True:
        current_modified = os.path.getmtime(target_file)
        if current_modified > last_modified:
            print(f"检测到{target_file}更改，重新生成PDF...")
            pdf_path = generate_preview_pdf()
            auto_open_pdf(pdf_path)
            last_modified = current_modified
        
        time.sleep(interval)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        this_file = __file__
        print(f"监视模式：当{this_file}发生更改时自动重新生成PDF...")
        pdf_path = generate_preview_pdf()
        auto_open_pdf(pdf_path)
        watch_for_changes(this_file)
    else:
        # 直接生成并打开PDF
        pdf_path = generate_preview_pdf()
        auto_open_pdf(pdf_path)
        print(f"已生成PDF：{pdf_path}")
