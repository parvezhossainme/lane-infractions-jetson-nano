import os
import sys

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lane_detection_hough import process_image


def test_process_image_preserves_shape_and_modifies_pixels():
    image = np.zeros((240, 320, 3), dtype=np.uint8)

    cv2.line(image, (40, 239), (140, 120), (255, 255, 255), 6)
    cv2.line(image, (280, 239), (180, 120), (255, 255, 255), 6)

    output = process_image(image)

    assert output.shape == image.shape
    assert output.dtype == image.dtype
    assert not np.array_equal(output, image)
