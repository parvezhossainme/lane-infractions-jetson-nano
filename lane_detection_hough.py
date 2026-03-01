import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines):
    img = np.copy(img)
    blank_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(blank_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    img = cv2.addWeighted(img, 0.8, blank_image, 1, 0.0)
    return img


def process_image(image):
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(gray_image, 100, 200)
    cropped_image = region_of_interest(
        canny_image,
        np.array([region_of_interest_vertices], np.int32),
    )

    lines = cv2.HoughLinesP(
        cropped_image,
        rho=2,
        theta=np.pi / 180,
        threshold=50,
        lines=np.array([]),
        minLineLength=40,
        maxLineGap=100,
    )

    image_with_lines = draw_lines(image, lines)
    return image_with_lines


def main():
    parser = argparse.ArgumentParser(description="Hough line lane detection demo")
    parser.add_argument("--input", default="test_image.jpg", help="Input image path")
    parser.add_argument("--output", default="outputs/tests/hough_lane_result.jpg", help="Output image path")
    parser.add_argument("--no-show", action="store_true", help="Disable matplotlib preview")
    args = parser.parse_args()

    image_bgr = cv2.imread(args.input)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read input image: {args.input}")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    processed_image = process_image(image_rgb)

    output_bgr = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(args.output, output_bgr)

    if not args.no_show:
        plt.imshow(processed_image)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    main()
