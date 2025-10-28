import os
from pdf2image import convert_from_path
from oci.addons.adk import Toolkit, tool
from PIL import Image

@tool
def convert_pdf_to_png(pdf_path, output_dir):
    """
    a tool to convert pdf to text
    :param pdf_path:
    :param output_dir:
    :return:
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert all pages to images (each page = 1 PNG)
    Image.MAX_IMAGE_PIXELS = None
    pages = convert_from_path(pdf_path, dpi=1024)

    for i, page in enumerate(pages):
        output_path = os.path.join(output_dir, f"page_{i+1}.png")
        page.save(output_path, "PNG")
        print(f"Saved: {output_path}")

def test_case():
    # Example usage
    pdf_file = "images/N5.pdf"  # Path to your image-based PDF
    output_folder = "images/output_images"
    convert_pdf_to_png(pdf_file, output_folder)

if __name__ == "__main__":
    test_case()
