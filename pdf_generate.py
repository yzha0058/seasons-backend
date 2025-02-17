from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from flask import Flask, send_file
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import io

# Register the font (adjust the path based on your system)
FONT_PATH = "font/SourceHanSans-VF.ttf.ttc"  # Common path on Linux
pdfmetrics.registerFont(TTFont("SourceHanSans", FONT_PATH))  # Register font

def generate_pdf(result):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    pdf.setTitle("Face Analysis Report")
    pdf.setFont("SourceHanSans", 16)

    x_start, y_start = 50, 800

    def draw_section(title, data):
        """ Helper function to write a section in the PDF with Chinese support """
        nonlocal y_start
        pdf.setFont("SourceHanSans", 14)
        pdf.drawString(x_start, y_start, title)
        y_start -= 20

        pdf.setFont("SourceHanSans", 12)
        for key, value in data.items():
            if isinstance(value, list):  # Convert list to string
                value = ", ".join(value)
            elif isinstance(value, dict):  # Convert dict to readable format
                value = ", ".join([f"{k}: {v}" for k, v in value.items()])

            pdf.drawString(x_start + 20, y_start, f"{key}: {value}")
            y_start -= 20
            if y_start < 50:  # Avoid writing out of bounds
                pdf.showPage()
                pdf.setFont("SourceHanSans", 12)
                y_start = 800

        y_start -= 10

    # Draw sections using example data
    draw_section("脸型分析", result["Face_shape"])
    draw_section("脸型详细信息", result["Face_shape_info"])
    draw_section("唇部详细信息", result["Lips_detailed_info"])
    draw_section("眼部详细信息", result["eye_detailed_info"])
    draw_section("眼型分析", result["eye_shape"])
    draw_section("唇型分析", result["lip_shape"])
    draw_section("鼻部详细信息", result["nose_detailed_info"])
    draw_section("鼻型分析", result["nose_shape"])

    pdf.showPage()
    pdf.save()

    buffer.seek(0)
    return buffer