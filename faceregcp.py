import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Streamlit App Title
st.title("Face Detection App")
st.markdown("<h6 style = 'top_margin: 0rem; color: #F2921D'>Built by LOLA</h6>", unsafe_allow_html = True)
st.write("Upload an image and adjust detection settings.")

# User instructions
st.markdown("""
### How to Use:
1. Upload an image.
2. Adjust the parameters (`scaleFactor` and `minNeighbors`) for better detection.
3. Choose the rectangle color for detected faces.
4. Save the output image if needed.
""")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# Parameters selection
scale_factor = st.slider("Scale Factor", 1.1, 2.0, 1.2, 0.1)
min_neighbors = st.slider("Min Neighbors", 3, 10, 5, 1)
rect_color = st.color_picker("Choose Rectangle Color", "#FF0000")  # Default: Red

# Convert hex color to BGR
hex_color = rect_color.lstrip('#')
bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # Convert to BGR

if uploaded_file is not None:
    # Convert to OpenCV format
    image = Image.open(uploaded_file)
    image_cv = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors)
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), bgr_color, 2)
    
    # Display result
    st.image(image_cv, caption="Detected Faces", use_column_width=True)
    
    # Save option
    save_option = st.button("Save Image with Detected Faces")
    if save_option:
        save_path = "detected_faces.jpg"
        cv2.imwrite(save_path, cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR))
        st.success(f"Image saved as {save_path}")
