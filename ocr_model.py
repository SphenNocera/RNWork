import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

files = [os.path.abspath(os.path.join(os.getcwd(), f"docs/acord_{number}.pdf")) for number in range( 1, 8 )]
page_number = 0

def convert_pdf_to_image(file: str) -> list[np.ndarray]:
    """
    Converts a pdf file into a numpy array of RGB values.
    """

    images = convert_from_path(file)
    images = get_image_data_for_opencv(images, cv.COLOR_BGR2GRAY)
    return images


def get_image_data_for_opencv(images: list, conversion_type: int) -> list[np.ndarray]:
    """
    Returns images as a list of numpy arrays in the format for opencv to read. the numpy arrays define the rgb values for the images.
    """

    # must convert to numpy array (rgb format) for opencv preprocessing
    converted_images = [
        cv.cvtColor(np.array(image), conversion_type) for image in images
    ]

    imgs = converted_images

    return imgs


def get_confidence_level_per_word(image: np.ndarray) -> dict:
    """
    Returns a dictionary with each word and its corresponding confidence level
    """
    image_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    word_to_conf = {}
    for index in range(len(image_data["conf"])):
        word_to_conf.update({image_data["text"][index]: image_data["conf"][index]})

    return word_to_conf


def get_average_confidence_level(images: list[np.ndarray]) -> list[float]:
    """
    Returns the average of every word and its corresponding confidence level.
    """
    if not isinstance(images, list):
        images = [images]

    total = []
    for image in images:
        word_conf_dict = get_confidence_level_per_word(image)
        total.append(sum(word_conf_dict.values()) / len(word_conf_dict))

    return total


def draw_bounding_boxes(
    images: list[np.ndarray], max_confidence_level: int = 75
) -> None:
    """
    Draws bounding boxes around each of the words tesseract has found where its confidence level is LOWER than max_confidence_level.
    max_confidence_level set to 100 will draw all bounding boxes.
    """

    if not isinstance(images, list):
        images = [images]

    for image in images:
        image_data = pytesseract.image_to_data(
            image, output_type=pytesseract.Output.DICT
        )
        n_boxes = len(image_data["level"])
        for i in range(n_boxes):
            conf = image_data["conf"][i]
            if conf > 0 and conf < max_confidence_level:
                (x, y, w, h) = (
                    image_data["left"][i],
                    image_data["top"][i],
                    image_data["width"][i],
                    image_data["height"][i],
                )
                cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        Image.fromarray(image).show()


def erode_image(images: list[np.ndarray], kernel_size: int = 3) -> list[np.ndarray]:
    """
    Returns the eroded images in order to make the text more bold. For more info read here: <https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm>
    """

    if not isinstance(images, list):
        images = [images]

    modified_images = []
    for image in images:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)

        modified_images.append(cv.erode(image, kernel, iterations=1))

    return modified_images


def scale_image(images: list[np.ndarray], amount: float = 1.4) -> list[np.ndarray]:
    """
    Returns the scaled images in order to make the text larger. For more info read here: <https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm>
    """

    if not isinstance(images, list):
        images = [images]

    modified_images = []
    for image in images:
        w, h = image.shape[:2]

        modified_images.append(cv.resize(image, (int(w * amount), int(h * amount))))

    return modified_images


def sharpen_image(images: list[np.ndarray]):

    if not isinstance(images, list):
        images = [images]

    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    modified_images = []
    for image in images:
        modified_images.append(cv.filter2D(image, -1, kernel))
    return modified_images


def blur_image(images: list[np.ndarray]):

    if not isinstance(images, list):
        images = [images]

    modified_images = []
    for image in images:
        modified_images.append(cv.GaussianBlur(image, (5, 5), 1))
    return modified_images


def denoise_image(images: list[np.ndarray]):

    if not isinstance(images, list):
        images = [images]

    modified_images = []
    for image in images:
        modified_images.append(cv.fastNlMeansDenoising(image, -1))
    return modified_images


def apply_threshold(images: list[np.ndarray], min_threshold: int):

    if not isinstance(images, list):
        images = [images]

    modified_images = []
    for image in images:
        modified_images.append(
            cv.threshold(image, min_threshold, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[
                1
            ]
        )
    return modified_images

def get_lines_in_image(image):
    if len(image.shape) != 2:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image

    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -2)
    
    hor = np.copy(bw)
    vert = np.copy(bw)

    cols = hor.shape[1]
    rows = vert.shape[0]
    h_size = cols // 30
    v_size = rows // 30

    hor_structure = cv.getStructuringElement(cv.MORPH_RECT, (h_size, 1))
    vert_structure = cv.getStructuringElement(cv.MORPH_RECT, (1, v_size))

    hor = cv.morphologyEx(hor, cv.MORPH_OPEN, hor_structure)
    vert = cv.morphologyEx(vert, cv.MORPH_OPEN, vert_structure)

    all_lines = cv.add(hor, vert)

    final = cv.add(image, vert)
    return final

def plot_averages(avgs1, avgs2, avgs3):
    modifications = ("base", "scaled (1.4x)", "scaled & eroded")
    width = .25
    x = np.arange(len(avgs1))

    fig, ax = plt.subplots(layout = "constrained")

    ax.bar(x, avgs1, width, data=avgs1, label=modifications[0])
    ax.bar(x+width,avgs2, width, data=avgs2, label = modifications[1])
    ax.bar(x+width*2,avgs3, width, data=avgs3, label = modifications[2])

    ax.set_xticklabels([f"acord_{file_number + 1}" for file_number in range(len(avgs1))])
    ax.set_xticks(x + width/3)
    ax.legend(loc="upper left", ncols=3)

    plt.show()


base_images = convert_pdf_to_image(files[2])[0]
lines_gone = get_lines_in_image(base_images)
Image.fromarray(lines_gone).show()
print(get_average_confidence_level(base_images))
print(get_average_confidence_level(lines_gone))
#Next step, try maybe inpainting for correction


