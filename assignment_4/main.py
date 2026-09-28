import os
import cv2
import numpy as np

REFERENCE_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "reference_img.png")
ALIGN_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "align_this.jpg")
SOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "solutions")

def detect_harris_corners(reference_image, max_corners=200, quality_level=0.01, min_distance=10):
    gray_img = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
    gray_img = np.float32(gray_img)

    corners = cv2.goodFeaturesToTrack(
        gray_img,
        maxCorners=0,
        qualityLevel=quality_level,
        minDistance=min_distance,
        useHarrisDetector=True,
        k=0.04,
    )

    detected_img = reference_image.copy()
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(detected_img, (int(x), int(y)), 3, (0, 0, 255), -1)

    return detected_img

def align_img (image_to_align, reference_image, max_features, good_match_precent):
    gray_alignment = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(max_features)
    alignment_keypoints, alignment_descriptors = orb.detectAndCompute(gray_alignment, None)
    ref_keypoints, ref_descriptors = orb.detectAndCompute(ref_gray, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = list(matcher.match(alignment_descriptors, ref_descriptors, None))

    matches = sorted(matches, key=lambda x: x.distance, reverse=False)
    match_count = int(len(matches) * good_match_precent)
    matches = matches[:match_count]

    matched_img = cv2.drawMatches(
        image_to_align, alignment_keypoints, reference_image,
        ref_keypoints, matches, None
    )

    alignment_points = np.zeros((len(matches), 2), dtype=np.float32)
    ref_points = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        alignment_points[i, :] = alignment_keypoints[match.queryIdx].pt
        ref_points[i, :] = ref_keypoints[match.trainIdx].pt

    homography, mask = cv2.findHomography(alignment_points, ref_points, cv2.RANSAC)

    height, width, _ = reference_image.shape
    aligned_img = cv2.warpPerspective(image_to_align, homography, (width, height))

    return aligned_img, matched_img

def main():
    reference_image = cv2.imread(REFERENCE_IMAGE_PATH)
    if reference_image is None:
        raise FileNotFoundError("Reference image not found")
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)

    image_to_align = cv2.imread(ALIGN_IMAGE_PATH)
    if image_to_align is None:
        raise FileNotFoundError("Alignment image not found")


    harris_corners_detected_img = detect_harris_corners(reference_image)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "harris.png"), harris_corners_detected_img)
    print("Harris corners image saved")

    aligned_img, matched_img = align_img(image_to_align,
                                         reference_image,
                                         max_features=1500,
                                         good_match_precent=0.15
                                         )
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "aligned.png"), aligned_img)
    cv2.imwrite(os.path.join(SOLUTIONS_DIR, "matches.png"), matched_img)
    print("Aligned and matched image saved")


if __name__ == "__main__":
    main()