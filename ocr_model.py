import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import re
import cv2

path_to_file = "docs/document-1.pdf"

def convert_pdf_to_image(file):
    images = convert_from_path(path_to_file)
    return images

# example function to extract info from the text
def extract_email(txt):
    return re.search(".*@.*\.com", txt, re.IGNORECASE).group(0)

def get_confidence_level_per_word(tesseract_data):
    word_to_conf = {}
    for index in range(len(tesseract_data["conf"])):
        word_to_conf.update({tesseract_data["text"][index]: tesseract_data["conf"][index]})
    
    return word_to_conf

def get_average_confidence_level(word_conf_dict: dict):
    total = sum(word_conf_dict.values())
    return total / len(word_conf_dict)


images = convert_pdf_to_image(path_to_file)
first_page_image = images[1]

first_page_image.show()

page_data = pytesseract.image_to_data(first_page_image, output_type=pytesseract.Output.DICT)

# print(page_data["text"])
confidence_per_word = get_confidence_level_per_word(page_data)
print(confidence_per_word)
avg_confidence = get_average_confidence_level(confidence_per_word)

print(avg_confidence)

