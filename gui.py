"""import tkinter as tk
from PIL import Image, ImageTk
import cv2
import threading
import time

class DetectionDashboard:
    def __init__(self, detector):
        self.detector = detector
        self.window = tk.Tk()
        self.window.title("Zero-Shot Object Detection")
        self.label = tk.Label(self.window)
        self.label.pack()
        self.video = cv2.VideoCapture(0)
        self.running = True
        self.fps_label = tk.Label(self.window, text="FPS: ")
        self.fps_label.pack()

    def update_frame(self):
        last_time = time.time()
        while self.running:
            ret, frame = self.video.read()
            if not ret: break

            detections = self.detector.detect(frame)
            for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
                if score > 0.1:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label_text = self.detector.prompts[label]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{label_text} {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Display in Tkinter
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(image))
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)

            # FPS
            now = time.time()
            fps = 1 / (now - last_time)
            last_time = now
            self.fps_label.config(text=f"FPS: {fps:.1f}")

    def run(self):
        thread = threading.Thread(target=self.update_frame)
        thread.daemon = True
        thread.start()
        self.window.mainloop()
        self.running = False
        self.video.release()"""

from PIL import Image, ImageTk
import cv2
import threading
import time
import tkinter as tk
import tkinter.filedialog as fd  # Add this to your imports

class DetectionDashboard:
    def __init__(self, detector):
        self.detector = detector
        self.prompt_history = []

        self.window = tk.Tk()
        self.window.title("Zero-Shot Object Detection")

        # Video feed display
        self.label = tk.Label(self.window)
        self.label.pack()

        # Prompts editor
        self.prompt_text = tk.Text(self.window, height=5, width=60)
        self.prompt_text.insert(tk.END, ", ".join(self.detector.prompts))
        self.prompt_text.pack()

        # Buttons Frame
        button_frame = tk.Frame(self.window)
        button_frame.pack()

        self.update_button = tk.Button(button_frame, text="Update Prompts", command=self.update_prompts)
        self.update_button.pack(side=tk.LEFT, padx=5)

        self.undo_button = tk.Button(button_frame, text="Undo", command=self.undo_prompt)
        self.undo_button.pack(side=tk.LEFT, padx=5)

        self.save_button = tk.Button(button_frame, text="Save Prompts", command=self.save_prompts)
        self.save_button.pack(side=tk.LEFT, padx=5)

        self.load_button = tk.Button(button_frame, text="Load Prompts", command=self.load_prompts)
        self.load_button.pack(side=tk.LEFT, padx=5)

        # FPS label
        self.fps_label = tk.Label(self.window, text="FPS: ")
        self.fps_label.pack()

        self.video = cv2.VideoCapture(0)
        self.running = True

    def update_prompts(self):
        current_prompts = self.prompt_text.get("1.0", tk.END).strip()
        if current_prompts:
            self.prompt_history.append(current_prompts)
            prompts = [p.strip() for p in current_prompts.split(",") if p.strip()]
            self.detector.prompts = prompts

    def undo_prompt(self):
        if self.prompt_history:
            last_prompts = self.prompt_history.pop()
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert(tk.END, last_prompts)
            prompts = [p.strip() for p in last_prompts.split(",") if p.strip()]
            self.detector.prompts = prompts

    def save_prompts(self):
        file_path = fd.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt")])
        if file_path:
            prompts = self.prompt_text.get("1.0", tk.END).strip()
            with open(file_path, "w") as file:
                file.write(prompts)

    def load_prompts(self):
        file_path = fd.askopenfilename(filetypes=[("Text Files", "*.txt")])
        if file_path:
            with open(file_path, "r") as file:
                loaded_prompts = file.read().strip()
                self.prompt_history.append(self.prompt_text.get("1.0", tk.END).strip())  # Save current before loading
                self.prompt_text.delete("1.0", tk.END)
                self.prompt_text.insert(tk.END, loaded_prompts)
                prompts = [p.strip() for p in loaded_prompts.split(",") if p.strip()]
                self.detector.prompts = prompts

    def update_frame(self):
        last_time = time.time()
        while self.running:
            ret, frame = self.video.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
                if score > 0.2:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    label_text = self.detector.prompts[label]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label_text} {score:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(Image.fromarray(image))
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
    

            now = time.time()
            fps = 1 / (now - last_time)
            last_time = now
            self.fps_label.config(text=f"FPS: {fps:.1f}")
    

    def run(self):
        thread = threading.Thread(target=self.update_frame)
        thread.daemon = True
        thread.start()
        self.window.mainloop()
        self.running = False
        self.video.release()

