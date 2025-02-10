import cv2
import numpy as np

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.source, self.target)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Verilen noktaları perspektif dönüşümü uygular.
        """
        if len(points) != 0:
            reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
            transformed_points = cv2.perspectiveTransform(reshaped_points, self.matrix)
            return transformed_points.reshape(-1, 2)
        else:
            # Hata durumunda geçici değer döndürür
            return np.array([[1, 2], [5, 8]], dtype='float')
