import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(
    page_title="Vehicle Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------
# Custom CSS for better styling
# -----------------------
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem;
        font-size: 16px;
        border-radius: 5px;
    }
    .upload-text {
        text-align: center;
        color: #666;
    }
    h1 {
        color: #2C3E50;
        text-align: center;
    }
    h3 {
        color: #34495E;
    }
    .stats-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# Load model
# -----------------------
@st.cache_resource
def load_model():
    try:
        model = YOLO('last.pt')
        return model
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None

# -----------------------
# Process image and run inference
# -----------------------
def process_image(image, model, conf_threshold, img_size):
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    img_array = np.array(image)
    
    results = model.predict(
        img_array, 
        conf=conf_threshold,
        imgsz=img_size,
        verbose=False
    )
    
    # Annotated image is already RGB
    annotated_img = np.array(results[0].plot())
    
    return annotated_img, results[0]

# -----------------------
# Main app
# -----------------------
def main():
    st.title("Vehicle Detection System")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.01,
            max_value=1.0,
            value=0.25,
            step=0.01,
            help="Lower values detect more objects but may include false positives"
        )
        
        img_size = st.select_slider(
            "Image Size for Inference",
            options=[320, 416, 640, 800, 1024],
            value=640,
            help="Larger sizes may detect smaller objects better but are slower"
        )
        
        st.markdown("---")
        st.markdown("### About")
        st.info("This system detects 8 types of vehicles: Auto, Bus, Car, LCV, Motorcycle, Multiaxle, Tractor, and Truck.")
        
        # Show model info
        st.markdown("---")
        st.markdown("### Model Info")
        if os.path.exists('last.pt'):
            file_size = os.path.getsize('last.pt') / (1024 * 1024)
            st.success(f"Model loaded ({file_size:.2f} MB)")
        else:
            st.error("Model file 'last.pt' not found!")
        
        st.markdown("---")
        st.markdown("### Tips")
        st.markdown("""
        - Use clear images with visible vehicles
        - Try lowering confidence threshold if no detections
        - Larger images work better
        - Good lighting helps detection
        """)
    
    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model. Please check if 'last.pt' exists in the current directory.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        # Display original image
        with col1:
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
            
            width, height = image.size
            st.caption(f"Size: {width}x{height} | Mode: {image.mode}")
            if width < 640 or height < 640:
                st.warning("Image is small. Consider using a larger image for better detection.")
        
        # Process and display results
        with col2:
            st.subheader("Detection Results")
            with st.spinner("Detecting vehicles..."):
                try:
                    annotated_img, results = process_image(image, model, conf_threshold, img_size)
                    st.image(annotated_img, use_container_width=True)
                    
                    total_detections = len(results.boxes)
                    if total_detections > 0:
                        st.success(f"Detected {total_detections} vehicle(s)")
                    else:
                        st.info("No detections at current threshold")
                    
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    return
        
        # Detection statistics
        st.markdown("---")
        if len(results.boxes) > 0:
            st.subheader("Detection Statistics")
            class_counts = {}
            for box in results.boxes:
                class_id = int(box.cls.cpu().numpy())
                class_name = model.names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            st.markdown(f"**Total Vehicles Detected: {len(results.boxes)}**")
            
            cols = st.columns(min(4, len(class_counts)))
            for idx, (class_name, count) in enumerate(class_counts.items()):
                with cols[idx % len(cols)]:
                    st.metric(label=class_name.capitalize(), value=count)
            
            # Detailed table
            st.markdown("---")
            st.subheader("Detailed Detections")
            detection_data = []
            for idx, box in enumerate(results.boxes):
                class_id = int(box.cls.cpu().numpy())
                class_name = model.names[class_id]
                confidence = float(box.conf.cpu().numpy())
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                width_box = x2 - x1
                height_box = y2 - y1
                detection_data.append({
                    "No.": idx + 1,
                    "Vehicle": class_name.capitalize(),
                    "Confidence": f"{confidence:.1%}",
                    "Box Size": f"{int(width_box)}x{int(height_box)}"
                })
            st.table(detection_data)
        else:
            st.warning("No vehicles detected in this image.")
            with st.expander("Troubleshooting"):
                st.markdown("""
                **Why no detections?**
                
                1. **Lower the confidence threshold**
                2. **Image quality** - use larger images
                3. **Image content** - must contain vehicles
                4. **Lighting** - poor lighting affects detection
                5. **Try a different image**
                
                **Vehicle types this model can detect:**
                - Auto, Bus, Car, LCV, Motorcycle, Multiaxle, Tractor, Truck
                """)
    
    else:
        st.markdown("<p class='upload-text'>Please upload an image to begin detection</p>", unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("What to upload?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Good Images:**")
            st.markdown("- Clear vehicle photos")
            st.markdown("- Good lighting")
            st.markdown("- High resolution")
        with col2:
            st.markdown("**Vehicle Types:**")
            st.markdown("- Cars, Buses, Trucks")
            st.markdown("- Motorcycles, Auto-rickshaws")
            st.markdown("- Tractors, LCVs")
        with col3:
            st.markdown("**Tips:**")
            st.markdown("- Use traffic/road images")
            st.markdown("- Multiple vehicles OK")
            st.markdown("- Adjust settings in sidebar")

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    main()
