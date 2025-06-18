from detections import ObjectDetector
from gui import DetectionDashboard

if __name__ == "__main__":
  
    detector = ObjectDetector(enable_logging=True, log_format="csv")

    app = DetectionDashboard(detector)
    app.run()
