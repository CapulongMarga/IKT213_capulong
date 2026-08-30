import os
import cv2
import numpy as np

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "iris.png")
SOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "solutions")

def padding(image, border_width):
    padded_img = cv2.copyMakeBorder(
        image,
        border_width,
        border_width,
        border_width,
        border_width,
        cv2.BORDER_REFLECT,
    )
    return padded_img

def crop(image, x_0, x_1, y_0, y_1):
    cropped_img = image[y_0:y_1, x_0:x_1]
    return cropped_img

def resize(image, width, height):
    resized_img = cv2.resize(image, (width, height))
    return resized_img

def copy(image, emptyPictureArray):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]
    return emptyPictureArray

def grayscale(image):
    grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_img

def hsv(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_img

def hue_shifted(image, emptyPictureArray, hue):
    height, width, channels = image.shape
    for y in range(height):
        for x in range(width):
            for c in range(channels):
                value_shifted = int(image[y, x, c]) + hue
                emptyPictureArray[y, x, c] = value_shifted % 256
    return emptyPictureArray

def smoothing(image):
    smoothed_img = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    return smoothed_img

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated_img = cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("Rotation angle is invalid. Use 90 or 180 degrees.")
    return rotated_img

def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not load image from {IMAGE_PATH}.")
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)

    padded_img = padding(image, border_width=100)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "padded_image.png"), padded_img)
    print("Padded image saved.")

    height, width = image.shape[:2]
    cropped_img = crop(image, x_0=200, x_1=width - 130, y_0=200, y_1=height - 130)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "cropped_image.png"), cropped_img)
    print("Cropped image saved.")

    resized_img = resize(image, width=200, height=200)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "resized_image.png"), resized_img)
    print("Resized image saved.")

    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    copied_img = copy(image, emptyPictureArray)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "copied_image.png"), copied_img)
    print("Copied image saved.")

    grayscale_img = grayscale(image)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "grayscale_image.png"), grayscale_img)
    print("Grayscale image saved.")

    hsv_img = hsv(image)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "hsv_image.png"), hsv_img)
    print("HSV image saved.")

    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)
    hue_shifted_img = hue_shifted(image, emptyPictureArray, 50)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "hue_shifted_image.png"), hue_shifted_img)
    print("Hue shifted image saved.")

    smoothed_img = smoothing(image)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "smoothed_image.png"), smoothed_img)
    print("Smoothed image saved.")

    rotated_img = rotation(image, rotation_angle=180)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "rotated_image.png"), rotated_img)
    print("Rotated image saved.")

if __name__ == "__main__":
    main()

