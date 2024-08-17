import os.path

import cv2 as cv
import numpy as np

class VideoWriter:
    def __init__(self, file: str, frame_size: tuple[int, int]=(960, 640), fps: int=10):
        # Append correct file extension
        if not file.endswith('.mp4'):
            file += '.mp4'

        self.file = file
        self.frame_size = frame_size
        self.fps = fps

        self.fourcc = cv.VideoWriter_fourcc(*'avc1')

        # Create directory if necessary
        os.makedirs(os.path.dirname(self.file), exist_ok = True)

        self.released = False
        self._writer = cv.VideoWriter(
            self.file, self.fourcc, self.fps, self.frame_size
        )
    
    def __del__(self):
        self.release()
    
    def release(self):
        if not self.released:
            self._writer.release()
            self.released = True

    def write_frame(self, image: np.ndarray):
        if image.size != self.frame_size:
            image = cv.resize(image, self.frame_size)

        self._writer.write(image)