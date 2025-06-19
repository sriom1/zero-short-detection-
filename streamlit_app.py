import streamlit as st
from detections import ObjectDetector
import cv2
from PIL import Image
import numpy as np
import time

def process_image(detector, image):
    try:
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Convert to PIL Image for processor
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process with truncation and padding
        inputs = detector.processor(
            text=detector.prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Run detection using processed inputs
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
    except Exception as e:
        st.error(f"Error in processing image: {str(e)}")
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
    
    # Add custom prompt
    st.sidebar.subheader("Add Custom Prompt")
    new_custom_prompt = st.sidebar.text_input("Enter a new prompt")
    if st.sidebar.button("Add Prompt"):
        if new_custom_prompt:
            st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
            st.session_state.detector.prompts.append(new_custom_prompt)
            st.sidebar.success(f"Added prompt: {new_custom_prompt}")
            st.rerun()  # Changed from st.experimental_rerun()
        else:
            st.sidebar.warning("Please enter a prompt first")
    
    # Current prompts display with delete buttons
    st.sidebar.subheader("Current Prompts")
    for i, prompt in enumerate(st.session_state.detector.prompts):
        col1, col2 = st.sidebar.columns([3, 1])
        with col1:
            st.text(f"{i+1}. {prompt}")
        with col2:
            # Delete prompt button
            if st.button("Delete", key=f"delete_{i}"):
                st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
                st.session_state.detector.prompts.pop(i)
                st.rerun()  # Changed from st.experimental_rerun()
    
    # Bulk edit prompts
    st.sidebar.subheader("Bulk Edit Prompts")
    current_prompts = ", ".join(st.session_state.detector.prompts)
    new_prompts = st.sidebar.text_area("Edit all prompts (comma-separated)", value=current_prompts)
    
    # Update prompts button
    if st.sidebar.button("Update All Prompts"):
        st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
        prompts = [p.strip() for p in new_prompts.split(",") if p.strip()]
        st.session_state.detector.prompts = prompts
        st.sidebar.success("Prompts updated successfully!")
    
    # Undo prompt changes
    if st.sidebar.button("Undo Changes") and st.session_state.prompt_history:
        previous_prompts = st.session_state.prompt_history.pop()
        st.session_state.detector.prompts = previous_prompts
        st.rerun()  # Changed from st.experimental_rerun()

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
            st.rerun()  # Changed from st.experimental_rerun()
        except Exception as e:
            st.sidebar.error(f"Error loading prompts: {str(e)}")
    
    # Show prompt count
    st.sidebar.subheader("Statistics")
    st.sidebar.text(f"Total Prompts: {len(st.session_state.detector.prompts)}")
    
    # Reset to defaults
    if st.sidebar.button("Reset to Default Prompts"):
        if st.sidebar.button("Confirm Reset"):
            st.session_state.prompt_history.append(st.session_state.detector.prompts.copy())
            st.session_state.detector = ObjectDetector(enable_logging=True, log_format="csv")
            st.rerun()  # Changed from st.experimental_rerun()

    # Main content - Tabs for different input methods
    tab1, tab2 = st.tabs(["Live Detection", "Image Upload"])
    
    with tab1:
        st.header("Live Detection")
        
        # Add a start/stop button
        if 'is_detecting' not in st.session_state:
            st.session_state.is_detecting = False

        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("Start" if not st.session_state.is_detecting else "Stop"):
                st.session_state.is_detecting = not st.session_state.is_detecting
                if not st.session_state.is_detecting:
                    st.rerun()

        with col1:
            # Create placeholder for video feed
            video_placeholder = st.empty()
            fps_placeholder = st.empty()

        if st.session_state.is_detecting:
            try:
                # Initialize video capture
                cap = cv2.VideoCapture(0)
                
                # Check if camera opened successfully
                if not cap.isOpened():
                    st.error("Error: Could not open camera")
                    st.session_state.is_detecting = False
                    st.rerun()
                
                while st.session_state.is_detecting:
                    start_time = time.time()
                    
                    # Capture frame
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                    
                    # Process frame
                    try:
                        # Run detection
                        detections = st.session_state.detector.detect(frame)
                        
                        # Draw detections
                        for box, score, label in zip(detections["boxes"], detections["scores"], detections["labels"]):
                            if score > 0.2:
                                x1, y1, x2, y2 = map(int, box.tolist())
                                label_text = st.session_state.detector.prompts[label]
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f"{label_text} {score:.2f}", 
                                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.5, (0, 255, 0), 2)
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        # Display the frame
                        video_placeholder.image(frame_rgb, channels="RGB")
                        
                        # Calculate and display FPS
                        fps = 1.0 / (time.time() - start_time)
                        fps_placeholder.text(f"FPS: {fps:.1f}")
                        
                    except Exception as e:
                        st.error(f"Error in detection: {str(e)}")
                        break
                    
                    # Add a small delay to prevent high CPU usage
                    time.sleep(0.01)
                
                # Release camera when stopped
                cap.release()
                
            except Exception as e:
                st.error(f"Camera error: {str(e)}")
            finally:
                if 'cap' in locals():
                    cap.release()
    
    with tab2:
        st.header("Upload Image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Convert uploaded file to image
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if image is None:
                    st.error("Failed to decode image. Please try another file.")
                    return
                
                # Process and display image
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                with col2:
                    st.subheader("Detected Objects")
                    processed_image = process_image(st.session_state.detector, image.copy())
                    st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()