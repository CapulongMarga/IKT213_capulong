import os
import cv2
import numpy as np

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "lambo.png")
SOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "solutions")

def sobel_edge_detection(image):
    blurry_image = cv2.GaussianBlur(image, (3, 3), 0)
    detect_edges = cv2.Sobel(blurry_image, cv2.CV_64F, dx=1, dy=1, ksize=1)
    detect_edges = cv2.convertScaleAbs(detect_edges)
    return detect_edges

def canny_edge_detection(image, threshold_1, threshold_2):
    blurry_image = cv2.GaussianBlur(image, (3, 3), 0)
    canny_image = cv2.Canny(blurry_image, threshold_1, threshold_2)
    return canny_image

SHAPES_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "shapes.png")
SHAPES_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "shapes_template.jpg")

def template_match(image, template):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_height, template_width = gray_template.shape[:2]

    result = cv2.matchTemplate(gray_image, gray_template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    locations = np.where(result >= threshold)

    match_image = image.copy()
    for point in zip(*locations[::-1]):
        top_left = point
        bottom_right = (point[0] + template_width, point[1] + template_height)
        cv2.rectangle(match_image, top_left, bottom_right, (0, 0, 255), 2)
    return match_image

def resize(image, scale_factor: int, up_or_down: str):
    resized_image = image.copy()
    for _ in range(scale_factor):
        if up_or_down == "up":
            resized_image = cv2.pyrUp(resized_image)
        elif up_or_down == "down":
            resized_image = cv2.pyrDown(resized_image)
        else:
            raise ValueError("Invalid option")
    return resized_image

def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError("Could not find the image")
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)

    detect_edges = sobel_edge_detection(image)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "detected_edges_image.png"), detect_edges)
    print("Detected edges image saved")

    canny_image = canny_edge_detection(image, threshold_1=50, threshold_2=50)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "canny_edges_image.png"), canny_image)
    print("Canny edges image saved")

    shapes_image = cv2.imread(SHAPES_IMAGE_PATH)
    shapes_template = cv2.imread(SHAPES_TEMPLATE_PATH)
    if shapes_image is None or shapes_template is None:
        raise FileNotFoundError("Could not find the shapes image or template image")
    match_image = template_match(shapes_image, shapes_template)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "match_image.png"), match_image)
    print("Match image saved")

    resized_image = resize(image, scale_factor=2, up_or_down="up")
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "resized_image.png"), resized_image)
    print("Resized image saved")

if __name__ == "__main__":
    main()

