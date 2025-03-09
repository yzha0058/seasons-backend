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

    x_start, y_start = 50, 800
    img_x, img_y = 350, 350  # Position for the images
    img_width, img_height = 200, 300  # Image size
    current_page = 1  # Track the current page number

    face_img_path = decode_base64_image(face_image)
    body_img_path = decode_base64_image(body_image)

    # ✅ Ensure face image is added at the beginning of the first page
    if face_img_path:
        pdf.drawImage(face_img_path, img_x, img_y, img_width, img_height)

    def draw_section(title, data, is_face_section=True):
        """ Draws a section and ensures images are added on the correct pages """
        nonlocal y_start, current_page
        
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

            if y_start < 50:  # Page break
                pdf.showPage()
                pdf.setFont("SourceHanSans", 12)
                y_start = 800
                current_page += 1  # Increment page number
                
                # ✅ Ensure face image is added on every new page of face analysis
                if is_face_section and face_img_path:
                    pdf.drawImage(face_img_path, img_x, img_y, img_width, img_height)

    # Draw Face Analysis Sections (Face image should appear on all face pages)
    draw_section("脸型详细信息", face_info["Face_shape_info"])
    draw_section("唇部详细信息", face_info["Lips_detailed_info"])
    draw_section("眼部详细信息", face_info["eye_detailed_info"])
    draw_section("眼型分析", face_info["eye_shape"])
    draw_section("唇型分析", face_info["lip_shape"])
    draw_section("鼻部详细信息", face_info["nose_detailed_info"])
    draw_section("鼻型分析", face_info["nose_shape"])

    # Move to a new page if not already on a new one for body analysis
    pdf.showPage()
    current_page += 1
    y_start = 800

    # ✅ Draw Body Analysis Section (Image should appear only here)
    draw_section("体型分析", body_info["body_detailed_info"], is_face_section=False)
    draw_section("三维分析", body_info["three_d_model_info"], is_face_section=False)

    # ✅ Ensure body image is placed correctly
    if body_img_path:
        pdf.drawImage(body_img_path, img_x, img_y, img_width, img_height)

    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    return fitz.open(stream=buffer.getvalue(), filetype="pdf")  # Convert to PyMuPDF doc
