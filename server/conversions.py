import base64

import cv2
import numpy as np


def b64str_to_cv2_image(b64_str: str) -> np.array:
    decoded_data = base64.b64decode(b64_str)
    np_data = np.frombuffer(decoded_data, np.uint8)

    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)


def cv2_image_to_b64str(image: np.array) -> str:
    _, img_encoded = cv2.imencode('.jpg', image)  # im_arr: image in Numpy one-dim array format.
    b64_str = base64.b64encode(img_encoded).decode('utf-8')
    return b64_str
