import sys
import os
import logging

import cv2
import numpy as np
from omegaconf import DictConfig

# logging setting
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s"
)
logger = logging.getLogger(__name__)

# define control key
Q_KEY = 0x71  # q key (end)
P_KEY = 0x70  # p key (pause)
D_KEY = 0x64  # d key (delete)
S_KEY = 0x73  # s key (save data and restart)


class DensityAnnotator:
    def __init__(self, cfg: DictConfig):
        logger.info(f"Loaded config: {cfg}")
        cv2.namedWindow("click annotation points")
        cv2.setMouseCallback("click annotation points", self.mouse_event)
        self.data_type = cfg.data_type
        self.sigma_pow = cfg.sigma_pow
        self.mouse_event_interval = cfg.mouse_event_interval

        # set frame information
        self.video = None
        self.frame = None
        self.frame_list = []
        self.width = None
        self.height = None
        self.features = None
        self.coordinate_matrix = None
        self.frame_num = 0

        # set file path
        self.input_file_path = cfg.path.input_file_path
        self.save_image_dir = cfg.path.save_image_dir
        self.save_annotated_dir = cfg.path.save_annotated_dir
        self.save_image_extension = cfg.save_image_extension
        self.save_annotated_coord_dir = f"{self.save_annotated_dir}/coord"
        self.save_annotated_density_dir = f"{self.save_annotated_dir}/dens"

    def run(self):
        """

        :return:
        """
        logger.info(f"Annotation Data Type: {self.data_type}")
        if self.data_type == "image":
            self.image_annotation()
        elif self.data_type == "movie":
            self.movie_annotation()
        else:
            logger.error(
                f"{self.data_type} is invalid data type. Please check config file."
            )
            sys.exit(1)

    def annotator_initialization(self):
        """

        :return:
        """
        self.width = self.frame.shape[1]
        self.height = self.frame.shape[0]

        #
        self.coordinate_matrix = np.zeros((self.width, self.height, 2), dtype="int64")
        for i in range(self.width):
            for j in range(self.height):
                self.coordinate_matrix[i][j] = [i, j]

    def image_annotation(self):
        """

        :return:
        """
        # load input image
        self.frame = self.load_image(self.input_file_path)
        self.frame_list.append(self.frame.copy())
        # initialize by frame information
        self.annotator_initialization()
        # display frame
        cv2.imshow("click annotation points", self.frame)

        # each key operation
        wait_interval = self.mouse_event_interval
        key = cv2.waitKey(wait_interval) & 0xFF
        if key == D_KEY:
            # delete the previous feature point
            self.delete_point()
        elif key == S_KEY:
            # save current annotated data and go to next frame
            self.save_annotated_data()
            wait_interval = self.mouse_event_interval
        # end processing
        cv2.destroyAllWindows()

    def movie_annotation(self):
        """

        :return:
        """
        # load input video data
        self.video = self.load_video(self.input_file_path)
        # load first frame and initialize by frame information
        ret, self.frame = self.video.read()
        self.annotator_initialization()

        # Read frames at regular intervals and annotate them.
        wait_interval = self.mouse_event_interval
        while ret:
            if wait_interval != 0:
                self.features = None
                self.frame_num += 1
                # display current frame
                cv2.imshow("click annotation points", self.frame)
                # load next frame and status
                ret, self.frame = self.video.read()

            # each key operation
            key = cv2.waitKey(wait_interval) & 0xFF
            if key == Q_KEY:
                # finish the annotation work
                break
            elif key == P_KEY:
                # pause current frame and start annotation
                interval = 0  # wait until the end of annotation
                self.frame_list.append(self.frame.copy())
                # save raw image
                cv2.imwrite(
                    f"{self.save_image_dir}/{self.frame_num}{self.save_image_extension}",
                    self.frame,
                )
            elif key == D_KEY:
                # delete the previous feature point
                self.delete_point()
            elif key == S_KEY:
                # save current annotated data and go to next frame
                self.save_annotated_data()
                wait_interval = self.mouse_event_interval
        # end processing
        cv2.destroyAllWindows()
        self.video.release()

    def mouse_event(self, event, x, y, flags, param):
        """
        select annotated point by left click of mouse

        :param event: mouse event
        :param x:
        :param y:
        :param flags:
        :param param:
        :return:
        """
        # other than left click
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # draw and add feature point
        cv2.circle(self.frame, (x, y), 4, (0, 0, 255), -1, 8, 0)
        self.add_point(x, y)
        cv2.imshow("click annotation points", self.frame)
        return

    def add_point(self, x, y):
        """
        add new feature point

        :param x:
        :param y:
        :return:
        """
        if self.features is None:
            self.features = np.array([[x, y]], np.uint16)
        else:
            self.features = np.append(self.features, [[x, y]], axis=0).astype(np.uint16)
        self.frame_list.append(self.frame.copy())

    def delete_point(self):
        """
        delete previous feature point

        :return:
        """
        if (self.features is not None) and (len(self.features) > 0):
            self.features = np.delete(self.features, -1, 0)
            self.frame_list.pop()
            self.frame = self.frame_list[-1].copy()
            cv2.imshow("click annotation points", self.frame)

    def calculate_gaussian_kernel(self) -> np.array:
        """
        calculate density map by gauss kernel
        :return:
        """
        kernel = np.zeros((self.width, self.height))

        for point in self.features:
            tmp_coord_matrix = np.array(self.coordinate_matrix)
            point_matrix = np.full((self.width, self.height, 2), point)
            diff_matrix = tmp_coord_matrix - point_matrix
            pow_matrix = diff_matrix * diff_matrix
            norm = pow_matrix[:, :, 0] + pow_matrix[:, :, 1]
            kernel += np.exp(-norm / (2 * self.sigma_pow))

        return kernel.T

    @staticmethod
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
            if ext in [".png", ".jpg"]:
                data_type = "image"
            elif ext in [".mp4"]:
                data_type = "movie"

        return data_type

    @staticmethod
    def load_image(path: str) -> np.array:
        """

        :param path:
        :return:
        """
        image = cv2.imread(path)
        return image

    @staticmethod
    def load_video(path: str) -> cv2.VideoCapture:
        video = cv2.VideoCapture(path)
        if not (video.isOpened()):
            logger.error("Error: Can not read video file")
            sys.exit(1)

        return video

    def save_annotated_data(self):
        """
        save coordinate and raw image. there are feature point information.
        :return:
        """
        if self.features is None:
            logger.info("None have been annotated.")
        else:
            # save image that added annotated point
            cv2.imwrite(
                f"{self.save_annotated_dir}/{self.frame_num}{self.save_image_extension}",
                self.frame,
            )
            # Save the coordinates of the annotated point
            np.savetxt(
                f"{self.save_annotated_coord_dir}/{self.frame_num}.csv",
                self.features,
                delimiter=",",
                fmt="%d",
            )
            # save annotated density map
            annotated_density = self.calculate_gaussian_kernel()
            np.save(f"{self.save_annotated_density_dir}/{self.frame_num}.npy")
            logger.info(f"Annotated and saved frame number: {self.frame_num}")
