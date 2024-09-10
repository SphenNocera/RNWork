import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import re

images = convert_from_path("document-1.pdf")

img = images[0]

img.show()

image_text = pytesseract.image_to_string(img)

# print(image_text)

def extract_email(txt):
    return re.search(".*@.*\.com", txt, re.IGNORECASE)

print(extract_email(image_text)) 