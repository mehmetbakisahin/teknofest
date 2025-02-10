import supervision as sv

class AnnotationManager:
    def __init__(self, video_fps: int, thickness: int = 1, text_scale: int = 0.5, text_padding:int = 1):
        self.bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=thickness)
        self.label_annotator = sv.LabelAnnotator(
            text_scale=text_scale,
            text_thickness=thickness,
            text_position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK,
            text_padding=text_padding
        )
        self.trace_annotator = sv.TraceAnnotator(
            thickness=thickness,
            trace_length=video_fps * 2,
            position=sv.Position.BOTTOM_CENTER,
            color_lookup=sv.ColorLookup.TRACK
        )

    def annotate(self, frame, detections, labels, polygon):
        """
        Frame üzerine bounding box, trace ve label bilgilerini çizerek geri döner.
        """
        annotated_frame = frame.copy()
        annotated_frame = sv.draw_polygon(annotated_frame, polygon=polygon, color=sv.Color.RED)
        annotated_frame = self.trace_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.bounding_box_annotator.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = self.label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        return annotated_frame
