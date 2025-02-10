import cv2

class VideoWriter:
    def __init__(self, output_path: str, fourcc: str, fps: int, frame_size: tuple):
        self.writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*fourcc),
            fps,
            frame_size
        )

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
