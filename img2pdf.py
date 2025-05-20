import os
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# 지원할 이미지 확장자
IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]

def images_to_pdf(img_dir, output_pdf):
    # 이미지 파일 리스트 정렬
    img_files = [f for f in os.listdir(img_dir)
                 if os.path.splitext(f)[1].lower() in IMG_EXTS]
    img_files.sort()

    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    font_size = 16
    margin = 30
    c.setFont("Helvetica", font_size)

    for img_name in img_files:
        img_path = os.path.join(img_dir, img_name)
        # 1. 이미지 이름 추가
        c.drawString(margin, height - margin - font_size, img_name)
        # 2. 이미지 추가
        try:
            img = Image.open(img_path).convert("RGB")
            # 비율 유지 최대 크기로 리사이즈
            max_w = int(width - 2*margin)
            max_h = int(height - 3*margin - font_size)
            img.thumbnail((max_w, max_h), Image.LANCZOS)
            img_w, img_h = img.size
            img_x = int((width - img_w) // 2)
            img_y = int(margin + 20)
            # BytesIO로 변환하여 drawImage에 전달
            from io import BytesIO
            img_buffer = BytesIO()
            img.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            c.drawImage(ImageReader(img_buffer), img_x, img_y, img_w, img_h)
        except Exception as e:
            c.drawString(margin, height//2, f"[이미지 로드 실패: {img_name}]")
            print(f"[ERROR] {img_name}: {e}")
        c.showPage()  # 다음 페이지로

    c.save()
    print(f"PDF 저장 완료: {output_pdf}")

if __name__ == "__main__":
    img_dir = "/home/byounggun/ultralytics/runs/detect/predict6"
    output_pdf = "predict6.pdf"
    images_to_pdf(img_dir, output_pdf)
