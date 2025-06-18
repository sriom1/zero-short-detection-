import streamlit as st
from detections import ObjectDetector
import cv2
from PIL import Image
import numpy as np
import time
import json

def process_image(detector, image):
    # Convert to RGB if needed
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    # Run detection
    detections = detector.detect(image)
    
    # Draw detections
    for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
        if score > 0.2:
            x1, y1, x2, y2 = map(int, box.tolist())
            label_text = detector.prompts[label]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label_text} {score:.2f}", (x1, y1-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return image

def load_prompt_history():
    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = []

def save_prompts_to_file(prompts, filename):
    with open(filename, 'w') as f:
        f.write(', '.join(prompts))

def load_prompts_from_file(file):
    content = file.read().decode()
    return [p.strip() for p in content.split(',') if p.strip()]

def main():
    st.title("Zero-Shot Object Detection")
    
    # Initialize detector and prompt history
    if 'detector' not in st.session_state:
        with st.spinner('Loading model...'):
            st.session_state.detector = ObjectDetector(enable_logging=True, log_format="csv")
    
    load_prompt_history()

    # Sidebar for prompts management
    st.sidebar.header("Prompts Management")
    
    # Current prompts display
    st.sidebar.subheader("Current Prompts")
    current_prompts = ", ".join(st.session_state.detector.prompts)
    new_prompts = st.sidebar.text_area("Edit prompts (comma-separated)", value=current_prompts)
    
    # Update prompts button
    if st.sidebar.button("Update Prompts"):
        # Save to history before updating
        st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
        prompts = [p.strip() for p in new_prompts.split(",") if p.strip()]
        st.session_state.detector.prompts = prompts
        st.sidebar.success("Prompts updated successfully!")
    
    # Undo prompt changes
    if st.sidebar.button("Undo Changes") and st.session_state.prompt_history:
        previous_prompts = st.session_state.prompt_history.pop()
        st.session_state.detector.prompts = previous_prompts
        st.experimental_rerun()
    
    # Save prompts to file
    if st.sidebar.button("Save Prompts"):
        try:
            save_prompts_to_file(st.session_state.detector.prompts, "saved_prompts.txt")
            st.sidebar.success("Prompts saved to saved_prompts.txt")
        except Exception as e:
            st.sidebar.error(f"Error saving prompts: {str(e)}")
    
    # Load prompts from file
    uploaded_file = st.sidebar.file_uploader("Load Prompts from File", type="txt")
    if uploaded_file is not None:
        try:
            loaded_prompts = load_prompts_from_file(uploaded_file)
            st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
            st.session_state.detector.prompts = loaded_prompts
            st.sidebar.success("Prompts loaded successfully!")
            st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading prompts: {str(e)}")
    
    # Prompt history display
    if st.session_state.prompt_history:
        st.sidebar.subheader("Prompt History")
        for i, hist_prompts in enumerate(st.session_state.prompt_history[-5:]):
            st.sidebar.text(f"{i+1}: {', '.join(hist_prompts)}")

    # Main content - Detection tabs
    tab1, tab2 = st.tabs(["Upload Image", "Live Camera"])
    
    with tab1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            # Convert uploaded file to image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Process and display image
            processed_image = process_image(st.session_state.detector, image)
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    
    with tab2:
        try:
            if 'video_capture' not in st.session_state:
                st.session_state.video_capture = cv2.VideoCapture(0)
            
            if not st.session_state.video_capture.isOpened():
                st.error("Failed to access webcam. Please try uploading an image instead.")
                return
            
            col1, col2 = st.columns([3, 1])
            with col1:
                frame_placeholder = st.empty()
            with col2:
                fps_placeholder = st.empty()
                stop_button = st.button("Stop Camera")
            
            while not stop_button:
                start_time = time.time()
                ret, frame = st.session_state.video_capture.read()
                
                if not ret:
                    st.error("Failed to capture video frame")
                    break
                
                processed_frame = process_image(st.session_state.detector, frame)
                frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
                
                fps = 1.0 / (time.time() - start_time)
                fps_placeholder.metric("FPS", f"{fps:.1f}")
                time.sleep(0.1)
                
        except Exception as e:
            st.error(f"Camera error: {str(e)}")
            st.info("Please try uploading an image instead.")
        finally:
            if 'video_capture' in st.session_state:
                st.session_state.video_capture.release()

if __name__ == "__main__":
    main()