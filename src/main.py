# main.py

import cv2
import numpy as np
import supervision as sv
from collections import defaultdict, deque
import logging

from config import (VIDEO_INPUT_PATH, VIDEO_OUTPUT_PATH, SOURCE, TARGET, VIDEO_SIZE, FPS, MODEL_PATH, MODEL_TRAINED_PATH
, PIXEL_TO_METER)
from detectors.yolov8_detector import YOLODetector
from tracking.byte_tracker import ByteTracker
from utils.view_transformer import ViewTransformer
from utils.video_io import VideoWriter
from utils.annotation_utils import AnnotationManager
from analytics.traffic_density_monitor import TrafficDensityMonitor
from logger.log_manager import LogManager  # Yeni: LogManager'ı içe aktarıyoruz


class VideoProcessor:
    def __init__(self):
        # Video bilgilerini al, fps belirle
        self.video_info = sv.VideoInfo.from_video_path(video_path=VIDEO_INPUT_PATH)
        self.fps = self.video_info.fps if self.video_info.fps else FPS

        # Tespit, takip, dönüşüm ve annotasyon sınıflarını başlat
        self.detector = YOLODetector(MODEL_TRAINED_PATH)
        self.tracker = ByteTracker(frame_rate=self.fps)
        self.view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
        self.annotation_manager = AnnotationManager(video_fps=self.fps)
        self.polygon_zone = sv.PolygonZone(polygon=SOURCE)

        # Hız hesaplaması için koordinatları saklamak amacıyla
        self.coordinates = defaultdict(lambda: deque(maxlen=int(self.fps)))

        # VideoWriter: Çıkış videosunu oluşturur
        self.video_writer = VideoWriter(
            output_path=VIDEO_OUTPUT_PATH,
            fourcc='mp4v',
            fps=int(self.fps),
            frame_size=VIDEO_SIZE
        )

        # Video frame üreticisi
        self.frame_generator = sv.get_video_frames_generator(source_path=VIDEO_INPUT_PATH)

        # Gelişmiş trafik yoğunluğu kontrolü için
        self.density_monitor = TrafficDensityMonitor(polygon=SOURCE, light_threshold=20, heavy_threshold=30)

        # LogManager: Loglama işlemleri için
        self.log_manager = LogManager(log_file="../data/logs/traffic_log.log", level=logging.INFO)
        self.frame_index = 0  # Frame sayacı

    def process(self):
        for frame in self.frame_generator:
            self.frame_index += 1

            # Frame boyutlandırma (orijinal örnekteki gibi)
            half_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

            # Tespit işlemi
            detections = self.detector.detect(half_frame)
            # Sadece belirlenen poligon alanındaki tespitleri kullan
            detections = detections[self.polygon_zone.trigger(detections)]

            # Toplam tespit sayısını loglayalım
            self.log_manager.log_detection(self.frame_index, len(detections))

            # Takip algoritması
            detections = self.tracker.update(detections)

            # Perspektif dönüşümü için alt orta nokta hesaplama
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = self.view_transformer.transform_points(points=points).astype(int)

            # Hız hesaplaması: Her tespit için koordinatları güncelle
            for tracker_id, point in zip(detections.tracker_id, points):
                _, y = point
                self.coordinates[tracker_id].append(y)

            # Label oluşturma: Takip numarası ve hız bilgisi
            labels = []
            for tracker_id in detections.tracker_id:
                if len(self.coordinates[tracker_id]) < self.fps / 2:
                    labels.append(f"#{tracker_id}")
                else:
                    coordinate_start = self.coordinates[tracker_id][-1]
                    coordinate_end = self.coordinates[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time_elapsed = len(self.coordinates[tracker_id]) / self.fps
                    speed = distance / time_elapsed * 3.6  # km/h
                    labels.append(f"#{tracker_id} {int(speed)} km/h")

            # Frame üzerine annotasyon ekleme (bounding box, trace, label)
            annotated_frame = self.annotation_manager.annotate(
                frame=half_frame,
                detections=detections,
                labels=labels,
                polygon=SOURCE
            )

            # Trafik yoğunluğu kontrolü ve annotasyonu
            density_category, count = self.density_monitor.check_density(detections)
            annotated_frame = self.density_monitor.annotate_frame(annotated_frame, density_category, count)

            # Loglama: Trafik yoğunluğu bilgisini logla
            self.log_manager.log_density(self.frame_index, density_category, count)

            # Videoya yaz ve ekranda göster
            self.video_writer.write(annotated_frame)
            cv2.imshow("Frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        self.video_writer.release()


if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process()
