import base64
import tempfile
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import io
import fitz
from PIL import Image

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

def decode_base64_image(base64_string, crop_center=True):
    """ Decodes base64 image, optionally crops the center, and saves it to a temporary file """
    try:
        # Decode base64
        image_data = base64.b64decode(base64_string.split(",")[1])  # Remove "data:image/png;base64,"
        img = Image.open(io.BytesIO(image_data))  # Open as a PIL image
        
        if crop_center:
            # Crop the middle part of the image
            width, height = img.size
            new_width, new_height = int(width * 0.6), int(height * 1.0)  # Keep 60% of the image
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            img = img.crop((left, top, right, bottom))  # Crop the middle portion
        
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(temp_file, format="PNG")  # Save the cropped image
        temp_file.close()
        
        return temp_file.name
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

def generate_pdf(face_info, body_info, face_image, body_image):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    pdf.setTitle("Face Analysis Report")
    
    page_width, page_height = A4
    x_start, y_start = 50, 800
    
    # 调整图片尺寸
    img_width, img_height = 220, 280  # 设置图片尺寸

    face_img_path = decode_base64_image(face_image)
    body_img_path = decode_base64_image(body_image)

    # === 🟦 背景框 ===
    box_x, box_y = 20, page_height - 300 # 背景框左下角坐标
    box_width, box_height = 310, 280  # 背景框宽度和高度
    radius = 10 # 圆角半径

    # 背景框颜色设置
    pdf.setFillColorRGB(0.85, 0.95, 0.9)
    pdf.roundRect(box_x, box_y, box_width, box_height, radius, fill=1, stroke=0)

    # === 🖋 SEASONS 标题 ===
    pdf.setFont("NotoSans-Black", 33)
    pdf.setFillColorRGB(0, 0, 0)
    title_x = box_x + 15
    title_y = box_y + box_height - 40

    # 绘制"SE"部分
    pdf.drawString(title_x, title_y, "SE")

    # 计算"SE"的宽度并获取绘制"A"的起始位置
    se_width = pdf.stringWidth("SE", "NotoSans-Black", 33)
    a_x = title_x + se_width + 2

    # 使用斜体字体绘制"A"
    pdf.setFont("NotoSans-BlackItalic", 35)
    pdf.drawString(a_x, title_y, "A")

    # 计算"A"的宽度并获取绘制"SONS"的起始位置
    a_width = pdf.stringWidth("A", "NotoSans-BlackItalic", 35)
    sons_x = a_x + a_width

    # 绘制"SONS"部分
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
        f"【脸型】 {face_info['Face_shape_info'].get('脸型判断结果', '')}",
        f"【眼型】 左眼: {face_info['eye_detailed_info'].get('左眼类型', '')} / 右眼: {face_info['eye_detailed_info'].get('右眼类型', '')}",
        f"【鼻型综合曲直】 {face_info['nose_detailed_info'].get('鼻型综合曲直', '')}",
        f"【唇形】 {face_info['lip_shape'].get('唇形', '')}",
        f"【腿型】 {body_info['three_d_model_info'].get('腿型', '')}",
        f"【身材类型】 {body_info['three_d_model_info'].get('身材类型', '')}",
    ]

    line_y = box_y + box_height - 130
    line_spacing = 26
    for line in summary_lines:
        pdf.drawString(box_x + 15, line_y, line)
        line_y -= line_spacing

    # === 📏 分隔线（1）===
    divider_y = box_y - 30
    pdf.setStrokeColorRGB(0.6, 0.85, 0.7)
    pdf.setLineWidth(2)
    pdf.line(20, divider_y, page_width - 20, divider_y)

    # === 📊 脸型详细信息（左列） ===
    # 标题
    pdf.setFont("NotoSans-Bold", 19)
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawString(x_start, divider_y - 40, "脸型详细信息")
    
    # 内容项
    pdf.setFont("NotoSans-Regular", 15)
    left_column_x = x_start - 15
    data_y = divider_y - 78
    
    face_shape_info = face_info["Face_shape_info"]
    data_items = [
        f"• 三庭比例: {face_shape_info.get('三庭比例', '')}",
        f"• 三线比例: {face_shape_info.get('三线比例', '')}",
        f"• 下巴形状: {face_shape_info.get('下巴形状', '')}",
        f"• 五眼比例: {face_shape_info.get('五眼比例', '')}",
        f"• 脸长和脸宽的比例: {face_shape_info.get('脸长和脸宽的比例', '')}"
    ]
    
    for item in data_items:
        pdf.drawString(left_column_x, data_y, item)
        data_y -= 27
    
    # === 📈 脸型分析结果（右列） ===
    pdf.setFont("NotoSans-Heavy", 15)
    pdf.setFillColorRGB(0.2, 0.6, 0.8)
    
    right_column_x = page_width / 2 + 20
    result_y = divider_y - 78
    
    result_items = [
        f"• 脸型判断结果: {face_shape_info.get('脸型判断结果', '')}",
        f"• 脸型曲直: {face_shape_info.get('脸型曲直', '')}",
        f"• 脸部量感: {face_shape_info.get('脸部量感', '')}",
        f"• 脸部风格: {face_shape_info.get('脸部风格', '')}"
    ]
    
    for item in result_items:
        pdf.drawString(right_column_x, result_y, item)
        result_y -= 30
    
    next_section_y = min(data_y, result_y) - 15
    
    # === 📏 分隔线（2）===
    divider_y = next_section_y
    pdf.setStrokeColorRGB(0.6, 0.85, 0.7)
    pdf.setLineWidth(2)
    pdf.line(20, divider_y, page_width - 20, divider_y)

    # === 👁 眼部详细信息（左列）+ 眼型分析（右列）===
    eye_section_y = divider_y - 40

    # 标题
    pdf.setFont("NotoSans-Bold", 19)
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawString(x_start, eye_section_y, "眼部详细信息")
    pdf.drawString(page_width / 2 + 40, eye_section_y, "眼型分析")

    # 眼部详细信息（左列）
    pdf.setFont("NotoSans-Regular", 15)
    eye_data_y = eye_section_y - 38
    left_column_x = x_start - 15

    # 左眼信息
    eye_info = face_info["eye_detailed_info"]
    pdf.drawString(left_column_x, eye_data_y, f"• 左眼内眼角角度: {eye_info.get('左眼内眼角角度', '')}")
    eye_data_y -= 27
    pdf.drawString(left_column_x, eye_data_y, f"• 左眼长高比例: {eye_info.get('左眼长高比例', '')}")
    eye_data_y -= 27
    pdf.drawString(left_column_x, eye_data_y, f"• 左眼特征: {eye_info.get('左眼特征', '')}")

    # 增加左眼特征和右眼内眼角角度之间的间距
    eye_data_y -= 40

    # 右眼信息
    pdf.drawString(left_column_x, eye_data_y, f"• 右眼内眼角角度: {eye_info.get('右眼内眼角角度', '')}")
    eye_data_y -= 27
    pdf.drawString(left_column_x, eye_data_y, f"• 右眼长高比例: {eye_info.get('右眼长高比例', '')}")
    eye_data_y -= 27
    pdf.drawString(left_column_x, eye_data_y, f"• 右眼特征: {eye_info.get('右眼特征', '')}")

    # 眼型分析（右列）- 黑色常规项目
    pdf.setFont("NotoSans-Regular", 15)
    pdf.setFillColorRGB(0, 0, 0)
    eye_result_y = eye_section_y - 38
    
    eye_shape = face_info["eye_detailed_info"]
    
    # 右眼分析
    pdf.drawString(right_column_x, eye_result_y, f"• 右眼曲直: {eye_shape.get('右眼曲直', '')}")
    eye_result_y -= 27
    pdf.drawString(right_column_x, eye_result_y, f"• 右眼类型: {eye_shape.get('右眼类型', '')}")

    # 增加右眼类型和左眼曲直之间的间距
    eye_result_y -= 40

    # 左眼分析
    pdf.drawString(right_column_x, eye_result_y, f"• 左眼曲直: {eye_shape.get('左眼曲直', '')}")
    eye_result_y -= 27
    pdf.drawString(right_column_x, eye_result_y, f"• 左眼类型: {eye_shape.get('左眼类型', '')}")

    # 增加左眼类型和眼型曲直综合之间的间距
    eye_result_y -= 40

    # 眼型分析 - 蓝色关键结论项
    pdf.setFont("NotoSans-Bold", 15)
    pdf.setFillColorRGB(0.2, 0.6, 0.8)

    # 综合结论
    pdf.drawString(right_column_x, eye_result_y, f"• 眼型曲直综合: {eye_shape.get('眼型曲直综合', '')}")
    eye_result_y -= 27
    pdf.drawString(right_column_x, eye_result_y, f"• 眼神: {eye_shape.get('眼神', '')}")

    # === 📏 分隔线（3）===
    divider_y = 20
    pdf.setStrokeColorRGB(0.6, 0.85, 0.7)
    pdf.setLineWidth(2)
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
    
    nose_info = face_info["nose_detailed_info"]
    
    # 鼻部详细信息项
    nose_data = [
        f"• 右脸山根: {nose_info.get('右脸山根', '')}",
        f"• 右鼻翼曲率: {nose_info.get('右鼻翼曲率', '')}",
        f"• 左脸山根: {nose_info.get('左脸山根', '')}",
        f"• 左鼻翼曲率: {nose_info.get('左鼻翼曲率', '')}",
        f"• 鼻孔比例: {nose_info.get('鼻孔比例', '')}"
    ]
    
    for item in nose_data:
        pdf.drawString(left_column_x, data_y, item)
        data_y -= 27
    
    # 鼻型分析
    pdf.setFont("NotoSans-Regular", 15)
    pdf.setFillColorRGB(0, 0, 0)
    nose_result_y = y_start
    
    nose_shape = face_info["nose_shape"]
    
    # 鼻型分析项
    nose_results = [
        f"• 鼻孔曲直: {nose_info.get('鼻孔曲直', '')}",
        f"• 山根曲直: {nose_info.get('山根曲直', '')}",
        f"• 鼻翼曲线综合判断: {nose_info.get('鼻翼曲线综合判断', '')}",
        f"• 鼻翼宽窄: {nose_info.get('鼻翼宽窄', '')}"
    ]
    
    for item in nose_results:
        pdf.drawString(right_column_x, nose_result_y, item)
        nose_result_y -= 30
    
    # 浅蓝色关键结论
    pdf.setFont("NotoSans-Bold", 15)
    pdf.setFillColorRGB(0.2, 0.6, 0.8)
    pdf.drawString(right_column_x, nose_result_y, f"• 鼻型综合曲直: {nose_info.get('鼻型综合曲直', '')}")
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
    
    lips_info = face_info["Lips_detailed_info"]
    
    # 唇部详细信息项
    lip_data = [
        f"• 上下唇比例: {lips_info.get('上下唇比例', '')}",
        # f"• 嘴角: {lips_info.get('嘴角', '')}",
        f"• 嘴角状态: {lips_info.get('嘴角状态', '')}",

        f"• 唇部数据: {lips_info.get('唇部数据', '')}",
        f"• 上唇: {lips_info.get('上唇', '')}",
        f"• 下唇: {lips_info.get('下唇', '')}"
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
        f"• 唇部数据: {lips_info.get('唇部数据', '')}",
        f"• 曲直结果: {lips_info.get('曲直结果', '')}",
        f"• 右嘴角倾斜度: {lips_info.get('右嘴角倾斜度', '')}",
        f"• 左嘴角倾斜度: {lips_info.get('左嘴角倾斜度', '')}",
    ]
    
    for item in lip_results:
        pdf.drawString(right_column_x, lip_result_y, item)
        lip_result_y -= 30
    
    # 浅蓝色关键结论
    pdf.setFont("NotoSans-Bold", 15)
    pdf.setFillColorRGB(0.2, 0.6, 0.8)
    pdf.drawString(right_column_x, lip_result_y, f"• 唇形: {face_info['lip_shape'].get('唇形', '')}")
    lip_result_y -= 30
    
    # 更新下一部分的位置
    next_section_y = min(data_y, lip_result_y) - 15
    
    # === 📏 分隔线（5）===
    divider_y = next_section_y
    pdf.setStrokeColorRGB(0.6, 0.85, 0.7)
    pdf.setLineWidth(2)
    pdf.line(20, divider_y, page_width - 20, divider_y)
    
    # === 体型分析和三维分析 ===
    pdf.setFont("NotoSans-Bold", 19)
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawString(page_width / 2 + 40, divider_y - 40, "体型分析")
    
    # 体型分析内容
    pdf.setFont("NotoSans-Regular", 15)
    data_y = divider_y - 78
    
    body_detailed_info = body_info["body_detailed_info"]
    
    # 体型分析项
    body_data = [
        f"• 上下半身比例: {body_detailed_info.get('上下半身比例', '')}",
        f"• 头肩比: {body_detailed_info.get('头肩比', '')}",
        f"• 头肩比判断: {body_detailed_info.get('头肩比判断', '')}"
    ]
    
    for item in body_data:
        pdf.drawString(right_column_x, data_y, item)
        data_y -= 27
    
    # 添加蓝色身材结论
    pdf.setFont("NotoSans-Regular", 15)
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawString(right_column_x, data_y, f"• 身材比例判断: {body_detailed_info.get('身材比例判断', '')}")
    data_y -= 50
    
    # 三维分析标题
    pdf.setFont("NotoSans-Bold", 19)
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawString(page_width / 2 + 40, data_y, "三维分析")
    data_y -= 38
    
    # 三维分析内容
    pdf.setFont("NotoSans-Regular", 15)
    pdf.setFillColorRGB(0, 0, 0)
    
    three_d_model_info = body_info["three_d_model_info"]
    
    # 三维分析项
    three_d_data = [
        f"• 三围比例: {three_d_model_info.get('三围比例', '')}"
    ]
    
    for item in three_d_data:
        pdf.drawString(right_column_x, data_y, item)
        data_y -= 27
    
    # 添加蓝色关键结论
    pdf.setFont("NotoSans-Bold", 15)
    pdf.setFillColorRGB(0.2, 0.6, 0.8)
    pdf.drawString(right_column_x, data_y, f"• 腿型: {three_d_model_info.get('腿型', '')}")
    data_y -= 27
    pdf.drawString(right_column_x, data_y, f"• 身材类型: {three_d_model_info.get('身材类型', '')}")
    
    # 绘制身体图像在左侧
    if body_img_path:
        img_x = 20
        img_y = 20
        pdf.drawImage(body_img_path, img_x, img_y, img_width, img_height)
    
    pdf.save()
    buffer.seek(0)
    
    return fitz.open(stream=buffer.getvalue(), filetype="pdf")
