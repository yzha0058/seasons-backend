import base64
import tempfile
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io
import fitz
from PIL import Image

FONT_PATH = "font/SourceHanSans-VF.ttf.ttc"
pdfmetrics.registerFont(TTFont("SourceHanSans", FONT_PATH))

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
    pdf.setFont("SourceHanSans", 16)

    page_width, page_height = A4
    x_start, y_start = 50, 800
    img_width, img_height = 200, 300

    face_img_path = decode_base64_image(face_image)
    body_img_path = decode_base64_image(body_image)

    # === ✅ Draw face image on top right ===
    if face_img_path:
        img_x = page_width - img_width - 50  # Right margin
        img_y = page_height - img_height - 50
        pdf.drawImage(face_img_path, img_x, img_y, img_width, img_height)

    # === ✅ Draw light green summary box with rounded corners ===
    box_x, box_y = 50, page_height - 250  # Top-left origin of the box
    box_width, box_height = 280, 180
    radius = 10

    # Light green color
    pdf.setFillColorRGB(0.8, 1, 0.8)
    pdf.roundRect(box_x, box_y, box_width, box_height, radius, fill=1, stroke=0)

    # === Title in the box ===
    pdf.setFont("SourceHanSans", 16)
    pdf.setFillColorRGB(0, 0, 0)
    pdf.drawString(box_x + 15, box_y + box_height - 25, "SEASONS 面部/体型分析")

    # === Add summary info ===
    pdf.setFont("SourceHanSans", 12)
    summary_lines = [
        f"[脸型] {face_info['Face_shape_info'].get('脸型判断结果', '')}",
        f"[眼型] 左眼: {face_info['eye_detailed_info'].get('左眼类型', '')} / 右眼:{face_info['eye_detailed_info'].get('右眼类型', '')}",
        f"[鼻型综合曲直] {face_info['nose_detailed_info'].get('鼻型综合曲直', '')}",
        f"[唇形] {face_info['lip_shape'].get('唇形', '')}",
        f"[腿型] {body_info['three_d_model_info'].get('腿型', '')}",
        f"[身材类型] {body_info['three_d_model_info'].get('身材类型', '')}",
    ]

    line_y = box_y + box_height - 50
    for line in summary_lines:
        pdf.drawString(box_x + 20, line_y, line)
        line_y -= 20

    # === Continue to other sections after summary ===
    y_start = box_y - 30  # Start drawing details below the summary box

    def draw_section(title, data):
        """ Draws a section and handles page breaks """
        nonlocal y_start
        pdf.setFont("SourceHanSans", 14)
        pdf.drawString(x_start, y_start, title)
        y_start -= 20

        pdf.setFont("SourceHanSans", 12)
        for key, value in data.items():
            if isinstance(value, list):
                value = ", ".join(value)
            elif isinstance(value, dict):
                value = ", ".join([f"{k}: {v}" for k, v in value.items()])
            pdf.drawString(x_start + 20, y_start, f"{key}: {value}")
            y_start -= 20

            if y_start < 50:
                pdf.showPage()
                y_start = 800
                pdf.setFont("SourceHanSans", 12)

    draw_section("脸型详细信息", face_info["Face_shape_info"])
    draw_section("唇部详细信息", face_info["Lips_detailed_info"])
    draw_section("眼部详细信息", face_info["eye_detailed_info"])
    draw_section("眼型分析", face_info["eye_shape"])
    draw_section("唇型分析", face_info["lip_shape"])
    draw_section("鼻部详细信息", face_info["nose_detailed_info"])
    draw_section("鼻型分析", face_info["nose_shape"])

    # New page for body analysis
    pdf.showPage()
    y_start = 800

    draw_section("体型分析", body_info["body_detailed_info"])
    draw_section("三维分析", body_info["three_d_model_info"])

    # Draw body image on body page
    if body_img_path:
        pdf.drawImage(body_img_path, img_x, img_y, img_width, img_height)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return fitz.open(stream=buffer.getvalue(), filetype="pdf")
