import os
import cv2

# Path to the iris image
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "iris-1.jpg")

# Given file in task 5
SOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "solutions")
CAMERA_OUTPUT_PATH = os.path.join(SOLUTIONS_DIR, "camera_outputs.txt")

def print_image_information(image):
    img_h, img_w, channels = image.shape
    img_size = image.size
    data_type = image.dtype

    print(f"height: {img_h}")
    print(f"width: {img_w}")
    print(f"channels: {channels}")
    print(f"size: {img_size}")
    print(f"data type: {data_type}")


def store_camera_info():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise IOError("The webcam could not be opened. Please make sure that it is connected.")

    camera_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    cap.release()

    os.makedirs(SOLUTIONS_DIR, exist_ok=True)

    with open(CAMERA_OUTPUT_PATH, "w") as f:
        f.write(f"fps: {int(camera_fps)}\n")
        f.write(f"height: {int(frame_h)}\n")
        f.write(f"width: {int(frame_w)}\n")

    print(f"The camera's info is stored in: {CAMERA_OUTPUT_PATH}")


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(
            f"Unable to load image from '{IMAGE_PATH}'. "
        )
    print_image_information(image)
    store_camera_info()

if __name__ == "__main__":
    main()