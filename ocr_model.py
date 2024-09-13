import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import re
import cv2 as cv
import os
import numpy as np

path_to_file = os.path.abspath(os.path.join(os.getcwd(), "docs/document-1.pdf"))

def convert_pdf_to_image(file):
    """
    Converts a pdf file into a PIL image type
    """

    images = convert_from_path(file)
    return images

# example function to extract info from the text
def extract_email(txt):
    return re.search(".*@.*\.com", txt, re.IGNORECASE).group(0)

def get_confidence_level_per_word(tesseract_data):
    word_to_conf = {}
    for index in range(len(tesseract_data["conf"])):
        word_to_conf.update({tesseract_data["text"][index]: tesseract_data["conf"][index]})
    
    return word_to_conf

def get_average_confidence_level(tesseract_data):
    word_conf_dict = get_confidence_level_per_word(tesseract_data)
    total = sum(word_conf_dict.values())
    return total / len(word_conf_dict)

def get_image_data_for_opencv(file):
    images = convert_pdf_to_image(file)

    # must convert to numpy array (rgb format) for opencv preprocessing
    converted_images = [cv.cvtColor(np.array(image), cv.COLOR_BGR2RGB) for image in images]

    img = converted_images[0]

    return img

    # cv.imshow("Original image", img)
    # cv.waitKey()

def compare_averages(tesseract_data1, tesseract_data2):
    avg1 = get_average_confidence_level(tesseract_data1)
    avg2 = get_average_confidence_level(tesseract_data2)

    print("base: " + str(avg1) + "\nInverted: " + str(avg2))

def invert_image(image):
    return cv.bitwise_not(image)
    

inverted_img_data = pytesseract.image_to_data(invert_image(get_image_data_for_opencv(path_to_file)), output_type=pytesseract.Output.DICT)
base_image_data = pytesseract.image_to_data(convert_pdf_to_image(path_to_file)[0], output_type=pytesseract.Output.DICT)

compare_averages(base_image_data, inverted_img_data)

