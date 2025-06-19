import streamlit as st
from detections import ObjectDetector
import cv2
from PIL import Image
import numpy as np
import time
import json

def process_frame(detector, frame):
    try:
        # Run detection
        detections = detector.detect(frame)
        
        # Draw detections
        for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
            if score > 0.2:
                x1, y1, x2, y2 = map(int, box.tolist())
                label_text = detector.prompts[label]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label_text} {score:.2f}", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                          0.5, (0, 255, 0), 2)
        return frame, detections
    except Exception as e:
        st.error(f"Detection error: {str(e)}")
        return frame, None

def main():
    st.title("Zero-Shot Object Detection")
    
    # Initialize detector and prompt history
    if 'detector' not in st.session_state:
        with st.spinner('Loading model...'):
            st.session_state.detector = ObjectDetector(enable_logging=True, log_format="csv")
    
    if 'prompt_history' not in st.session_state:
        st.session_state.prompt_history = []

    # Sidebar for prompt management
    st.sidebar.header("Prompt Management")

    # Current prompts display
    st.sidebar.subheader("Current Prompts")
    current_prompts = ", ".join(st.session_state.detector.prompts)
    new_prompts = st.sidebar.text_area("Edit prompts (comma-separated)", value=current_prompts)

    # Add new prompt
    new_prompt = st.sidebar.text_input("Add New Prompt")
    if st.sidebar.button("Add Prompt"):
        if new_prompt:
            st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
            st.session_state.detector.prompts.append(new_prompt)
            st.sidebar.success(f"Added: {new_prompt}")
            st.rerun()

    # Update all prompts
    if st.sidebar.button("Update All Prompts"):
        st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
        prompts = [p.strip() for p in new_prompts.split(",") if p.strip()]
        st.session_state.detector.prompts = prompts
        st.sidebar.success("Prompts updated successfully!")

    # Undo changes
    if st.sidebar.button("Undo Changes") and st.session_state.prompt_history:
        previous_prompts = st.session_state.prompt_history.pop()
        st.session_state.detector.prompts = previous_prompts
        st.rerun()

    # Save prompts to file
    if st.sidebar.button("Save Prompts"):
        try:
            with open("saved_prompts.txt", "w") as f:
                f.write(", ".join(st.session_state.detector.prompts))
            st.sidebar.success("Prompts saved to saved_prompts.txt")
        except Exception as e:
            st.sidebar.error(f"Error saving prompts: {str(e)}")

    # Load prompts from file
    uploaded_file = st.sidebar.file_uploader("Load Prompts from File", type="txt")
    if uploaded_file is not None:
        try:
            content = uploaded_file.read().decode()
            st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
            loaded_prompts = [p.strip() for p in content.split(",") if p.strip()]
            st.session_state.detector.prompts = loaded_prompts
            st.sidebar.success("Prompts loaded successfully!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading prompts: {str(e)}")

    # Show current prompts list
    st.sidebar.subheader("Active Prompts")
    for i, prompt in enumerate(st.session_state.detector.prompts):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.text(f"{i+1}. {prompt}")
        with col2:
            if st.button("Delete", key=f"delete_{i}"):
                st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
                st.session_state.detector.prompts.pop(i)
                st.rerun()

    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Live Detection")
        # Using st.camera_input for live camera feed
        camera_frame = st.camera_input("Camera")
        
        if camera_frame is not None:
            # Convert camera frame to OpenCV format
            bytes_data = camera_frame.getvalue()
            img_array = np.asarray(bytearray(bytes_data), dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            # Process frame and get detections
            start_time = time.time()
            processed_frame, detections = process_frame(st.session_state.detector, frame)
            fps = 1.0 / (time.time() - start_time)
            
            # Display processed frame
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))
            st.metric("FPS", f"{fps:.1f}")
    
    with col2:
        st.header("Detections")
        if 'last_detections' in st.session_state and st.session_state.last_detections:
            for entry in st.session_state.last_detections["detection_entries"][-5:]:
                st.markdown(f"""
                **Object**: {entry['object_detected']}  
                **Confidence**: {entry['confidence_score']:.2f}
                ---
                """)

    # Detection history in sidebar
    if st.sidebar.button("Show Full Detection History"):
        try:
            with open(st.session_state.detector.detection_file, 'r') as f:
                detection_history = json.load(f)
            
            st.sidebar.subheader("Recent Detections")
            for entry in detection_history[-10:]:
                st.sidebar.markdown(f"""
                **ID**: `{entry['detection_id'][:8]}`  
                **Object**: {entry['object_detected']}  
                **Confidence**: {entry['confidence_score']:.2f}  
                **Time**: {entry['timestamp']}
                ---
                """)
        except Exception as e:
            st.sidebar.error(f"Error loading detection history: {str(e)}")

if __name__ == "__main__":
    main()