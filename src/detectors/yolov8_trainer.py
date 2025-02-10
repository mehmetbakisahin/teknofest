# detectors/yolov8_trainer.py

from ultralytics import YOLO


class YOLOTrainer:
    def __init__(self, model_path: str = "yolov8n.pt"):
        """
        Args:
            model_path (str): Eğitim için kullanılacak önceden eğitilmiş model dosyasının yolu.
        """
        self.model = YOLO(model_path)

    def train(self, data: str, epochs: int = 50, batch: int = 16, imgsz: int = 640, **kwargs):
        """
        Modeli belirtilen parametrelerle eğitir.

        Args:
            data (str): Eğitim verilerini ve sınıf bilgilerini içeren YAML dosyasının yolu (ör. "data.yaml").
            epochs (int): Eğitim döngüsü (epoch) sayısı.
            batch (int): Batch size.
            imgsz (int): Eğitim sırasında kullanılacak resim boyutu.
            **kwargs: Ultralytics train metoduna aktarılacak diğer parametreler.

        Returns:
            results: Eğitim işleminin sonuçları.
        """
        results = self.model.train(data=data, epochs=epochs, batch=batch, imgsz=imgsz, **kwargs)
        return results
