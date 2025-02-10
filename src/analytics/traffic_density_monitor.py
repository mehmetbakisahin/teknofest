import cv2
import numpy as np
import supervision as sv


class TrafficDensityMonitor:
    def __init__(self, polygon: np.ndarray, light_threshold: int = 5, heavy_threshold: int = 10):
        """
        Args:
            polygon (np.ndarray): The region of interest defined by polygon coordinates to monitor density.
            light_threshold (int): If the vehicle count is below this value, the region is considered "Normal".
            heavy_threshold (int): If the vehicle count is between light_threshold and heavy_threshold, it is considered "Light Traffic".
                                   If it exceeds heavy_threshold, it is considered "Heavy Traffic".
        """
        self.polygon = polygon
        self.light_threshold = light_threshold
        self.heavy_threshold = heavy_threshold
        self.zone = sv.PolygonZone(polygon=polygon)

    def check_density(self, detections) -> (str, int):
        """
        Checks the number of vehicles within the defined polygon region based on given detections.

        Returns:
            tuple: (category, count)
                   category (str): "Normal", "Light Traffic", or "Heavy Traffic".
                   count (int): The number of vehicles in the region.
        """
        zone_detections = detections[self.zone.trigger(detections)]
        count = len(zone_detections)
        if count < self.light_threshold:
            category = "Normal"
        elif count < self.heavy_threshold:
            category = "Light Traffic"
        else:
            category = "Heavy Traffic"
        return category, count

    def annotate_frame(self, frame, category: str, count: int):
        """
        Adds a label on the frame to display the current traffic density status.

        Args:
            frame: The frame being processed.
            category (str): The traffic density category ("Normal", "Light Traffic", "Heavy Traffic").
            count (int): The number of vehicles in the region.

        Returns:
            frame: The updated frame with the traffic density label.
        """
        # Define color: Normal = Green, Light Traffic = Orange, Heavy Traffic = Red
        if category == "Normal":
            color = (0, 255, 0)
        elif category == "Light Traffic":
            color = (0, 165, 255)  # Orange tone
        else:  # "Heavy Traffic"
            color = (0, 0, 255)

        # Draw the polygon with the specified color
        frame = cv2.polylines(frame, [self.polygon.astype(np.int32)], isClosed=True, color=color, thickness=1)
        # Write the traffic density status on the frame
        cv2.putText(
            frame,
            f"Density: {category} ({count})",
            (25, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        return frame
