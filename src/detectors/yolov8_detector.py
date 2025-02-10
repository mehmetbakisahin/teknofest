from ultralytics import YOLO
import supervision as sv

class YOLODetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """
        Verilen frame üzerinde tespit işlemini gerçekleştirir.
        Dönüş: supervision.Detections nesnesi.
        """
        results = self.model(frame)
        detections = sv.Detections.from_ultralytics(results[0])
        return detections
