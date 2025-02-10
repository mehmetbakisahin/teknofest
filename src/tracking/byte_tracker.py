import supervision as sv

class ByteTracker:
    def __init__(self, frame_rate: int, activation_threshold: float = 0.5):
        self.byte_track = sv.ByteTrack(frame_rate=frame_rate, track_activation_threshold=activation_threshold)

    def update(self, detections):
        """
        Gelen tespit sonuçlarını takip algoritmasına sokar.
        Dönüş: güncellenmiş detections nesnesi.
        """
        return self.byte_track.update_with_detections(detections=detections)
