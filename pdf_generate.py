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
    buffer = io.BytesIO()  # Use an in-memory buffer
    pdf = canvas.Canvas(buffer, pagesize=A4)
    
    pdf.setTitle("Face Analysis Report")
    pdf.setFont("SourceHanSans", 16)

    # Set starting position
    x_start = 50
    y_start = 800
    
    pdf.setFont("SourceHanSans", 16)
    pdf.drawString(x_start, y_start, "Face Analysis Report")
    pdf.setFont("SourceHanSans", 12)
    
    y_start -= 30  # Move down

    def draw_section(title, data):
        """ Helper function to write a section in the PDF """
        nonlocal y_start
        pdf.setFont("SourceHanSans", 14)
        pdf.drawString(x_start, y_start, title)
        y_start -= 20
        
        pdf.setFont("SourceHanSans", 12)
        for key, value in data.items():
            pdf.drawString(x_start + 20, y_start, f"{key}: {value}")
            y_start -= 20
            if y_start < 50:  # Avoid writing out of bounds
                pdf.showPage()
                y_start = 800  # Reset for new page
        
        y_start -= 10  # Add spacing
    
   # Draw sections
    draw_section("身体分析结果", result["body_shape"])
    draw_section("详细身体信息", result["Body_detailed_info"])
    draw_section("三维模型分析", result["three_d_model"])
    draw_section("三维模型信息", result["three_d_model_info"])
    draw_section("面部体积分析", result["face_volume_analysis"])
    draw_section("面部体积信息", result["face_volume_info"])
    
    pdf.showPage()
    pdf.save()
    
    buffer.seek(0)
    return buffer