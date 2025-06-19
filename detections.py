import os, csv
import datetime
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import torch, cv2, numpy as np, time
from PIL import Image
from uuid import uuid4
import json
from pathlib import Path

class ObjectDetector:
    def __init__(self, prompts=None, use_torchscript=False, input_size=(320, 240), enable_logging=True, log_format="csv"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
        self.model = model.to(self.device)
        self.prompts = prompts or [
            "monitor", "keyboard", "mug",
            "a transluscent purple water bottle",
            "a transparent wireless computer mouse with a visible scroll wheel and internal components",
            "a thin cylindrical transparent blue pen"
        ]
        self.use_torchscript = use_torchscript
        """if use_torchscript:
            self.model = torch.jit.script(self.model)"""
        if use_torchscript:
            traced_model_path = "owlvit_traced.pt"
            example_inputs = self.processor(text=self.prompts,images=Image.new("RGB", (320, 240)), return_tensors="pt")["pixel_values"]

            traced_model = torch.jit.trace(self.model, (example_inputs.to(self.device),))
            traced_model.save(traced_model_path)
            self.model = torch.jit.load(traced_model_path).to(self.device)
            print("[TorchScript] Model traced and loaded.")


        self.input_size = input_size
        self.enable_logging = enable_logging
        self.log_format = log_format
        self.log_file = f"detections_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.{log_format}"
        print(f"[LOG] Saving detections to: {os.path.abspath(self.log_file)}")

        # Add detection results directory
        self.results_dir = Path("detection_results")
        self.results_dir.mkdir(exist_ok=True)
        self.detection_file = self.results_dir / "detections.json"
        
        # Initialize detection file if it doesn't exist
        if not self.detection_file.exists():
            with open(self.detection_file, 'w') as f:
                json.dump([], f)

        if self.enable_logging:
            self._prepare_log_file()

    def _prepare_log_file(self):
        if self.log_format == "csv":
            with open(self.log_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "label", "confidence", "x1", "y1", "x2", "y2"])
        elif self.log_format == "json":
            with open(self.log_file, mode="w") as f:
                json.dump([], f)

    def _log_detection(self, label, score, box):
        timestamp = datetime.datetime.now().isoformat()
        x1, y1, x2, y2 = map(int, box.tolist())
        if self.log_format == "csv":
            with open(self.log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, label, f"{score:.4f}", x1, y1, x2, y2])
        elif self.log_format == "json":
            with open(self.log_file, mode="r+") as f:
                data = json.load(f)
                data.append({
                    "timestamp": timestamp,
                    "label": label,
                    "confidence": float(f"{score:.4f}"),
                    "bbox": [x1, y1, x2, y2]
                })
                f.seek(0)
                json.dump(data, f, indent=2)

    def _log_detection_with_id(self, detections, frame_timestamp):
        """Log detections with unique IDs and confidence scores"""
        detection_data = []
        
        for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
            if score > 0.2:  # Only log detections above threshold
                detection_id = str(uuid4())
                label_text = self.prompts[label]
                x1, y1, x2, y2 = map(int, box.tolist())
                
                detection_entry = {
                    "detection_id": detection_id,
                    "timestamp": frame_timestamp,
                    "object_detected": label_text,
                    "confidence_score": float(f"{score:.4f}"),
                    "bounding_box": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    }
                }
                detection_data.append(detection_entry)
        
        # Read existing detections
        try:
            with open(self.detection_file, 'r') as f:
                existing_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_data = []
        
        # Append new detections
        existing_data.extend(detection_data)
        
        # Write back to file
        with open(self.detection_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        return detection_data

    def detect(self, frame):
        try:
            original_size = frame.shape[1], frame.shape[0]  # (width, height)
            resized_frame = cv2.resize(frame, self.input_size)
            image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            
            # Add padding and truncation
            inputs = self.processor(
                text=self.prompts, 
                images=image, 
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            result = self.processor.post_process(
                outputs, 
                target_sizes=torch.tensor([image.size[::-1]]).to(self.device)
            )[0]

            # Scale boxes back to original frame size
            scale_x = original_size[0] / self.input_size[0]
            scale_y = original_size[1] / self.input_size[1]
            for i, box in enumerate(result["boxes"]):
                box[0] *= scale_x
                box[2] *= scale_x
                box[1] *= scale_y
                box[3] *= scale_y

                if self.enable_logging and result["scores"][i] > 0.1:
                    label = self.prompts[result["labels"][i]]
                    score = result["scores"][i].item()
                    self._log_detection(label, score, box)

            # Add timestamp to detections
            frame_timestamp = datetime.datetime.now().isoformat()
            
            # Log detections with IDs
            detection_entries = self._log_detection_with_id(result, frame_timestamp)
            
            # Add detection entries to result for use in UI
            result["detection_entries"] = detection_entries
            
            return result
        except Exception as e:
            print(f"Detection error: {str(e)}")
            # Return empty result with same structure
            return {
                "boxes": torch.tensor([]),
                "scores": torch.tensor([]),
                "labels": torch.tensor([]),
                "detection_entries": []
            }
