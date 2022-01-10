import logging
import os
import sys

import cv2
import numpy as np

# logging setting
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)


def get_input_data_type(path: str) -> str:
    """
    Get the extension from the input data path and get the data processing format.
    The target format is images or videos.

    :param path: file path to be annotated
    :return: processing format of annotator
    """
    data_type = "invalid"
    if os.path.isfile(path):
        root, ext = os.path.splitext(path)
        if ext in [".png", ".jpg", ".jpeg"]:
            data_type = "image"
        elif ext in [".mp4", ".mov"]:
            data_type = "movie"

    logger.info(f"Input Data Type: {data_type}")

    return data_type


def load_image(path: str) -> np.array:
    """
    Loads image data from the input path and returns image in numpy array format.

    :param path: input image file path
    :return: loaded image
    """
    image = cv2.imread(path)
    if image is None:
        logger.error(
            f"Error: Can not read image file. Please check input file path. {path}"
        )
        sys.exit(1)
    logger.info(f"Loaded Image: {path}")

    return image


def load_video(path: str) -> cv2.VideoCapture:
    """
    Loads video data from the input path and returns video in cv2.VideoCapture format.

    :param path: input video file path
    :return: loaded video
    """
    video = cv2.VideoCapture(path)
    if not (video.isOpened()):
        logger.error(
            f"Error: Can not read video file. Please check input file path. {path}"
        )
        sys.exit(1)
    logger.info(f"Loaded Video: {path}")

    return video


def save_image(path: str, image: np.array) -> None:
    """
    Save the image data in numpy format in the target path.

    :param path: save path of image
    :param image: target image
    :return: None
    """
    cv2.imwrite(path, image)
    logger.info(f"Saved Image: {path}")


def save_coordinate(path: str, coordinate: np.array) -> None:
    """
    Save the coordinate data (x, y) in numpy format in the target path.

    :param path: save path of coordinate
    :param coordinate: coordinate of annotated points
    :return: None
    """
    np.savetxt(path, coordinate, delimiter=",", fmt="%d")
    logger.info(f"Saved Coordinate: {path}")


def save_density_map(path: str, density_map: np.array) -> None:
    """
    Save the density map data in numpy format in the target path.

    :param path: save path of density map
    :param density_map: annotated density map
    :return: None
    """
    np.save(path, density_map)
    logger.info(f"Save Density Map: {path}")
