# utils/log_manager.py

import logging
import os


class LogManager:
    def __init__(self, log_file: str = "traffic_log.log", level: int = logging.INFO):
        """
        LogManager sınıfı loglama işlemlerini yapılandırır.

        Args:
            log_file (str): Logların yazılacağı dosya yolu.
            level (int): Loglama seviyesidir (örn. logging.INFO, logging.DEBUG).
        """
        # Log dizininin var olduğundan emin olun
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Temel log konfigürasyonu
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("TrafficLogger")

    def log_density(self, frame_index: int, category: str, count: int):
        """
        Her frame için trafik yoğunluğu verilerini loglar.

        Args:
            frame_index (int): İşlenen frame numarası.
            category (str): Trafik yoğunluk kategorisi (Normal, Hafif Yoğun, Çok Yoğun).
            count (int): Poligon alanındaki araç sayısı.
        """
        self.logger.info(f"Frame {frame_index}: Trafik Yoğunluğu: {category} ({count} araç)")

    def log_detection(self, frame_index: int, total_detections: int):
        """
        Her frame için toplam tespit sayısını loglar.

        Args:
            frame_index (int): İşlenen frame numarası.
            total_detections (int): Tespit edilen araç sayısı.
        """
        self.logger.info(f"Frame {frame_index}: Toplam Tespit: {total_detections}")
