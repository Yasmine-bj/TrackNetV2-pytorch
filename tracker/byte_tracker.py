import supervision as sv

class ByteTrackerWrapper:
    """Encapsule sv.ByteTrack pour simplifier le reste du code."""

    def __init__(self):
        self._tracker = sv.ByteTrack()

    def update(self, detections: sv.Detections) -> sv.Detections:
        return self._tracker.update_with_detections(detections)
