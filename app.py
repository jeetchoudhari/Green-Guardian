import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import cv2

# Load your model
model = load_model('waste.h5')

# Set up the Streamlit app
st.title('Waste Classification App')
st.write('Upload an image and the model will predict if it is Wet Waste or Dry Waste.')

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        class_names = ['O', 'R']

        # Read the image file
        img = Image.open(uploaded_file)

        # Display the uploaded image
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Perform any necessary preprocessing (resize, normalize, etc.)
        img = img.resize((242, 208))
        
        # Convert the image to a numpy array
        img_array = np.array(img)
        
        # Make predictions using the model
        pred = model.predict(np.expand_dims(img_array, axis=0))
        
        # Get the predicted class
        predicted_class = class_names[np.argmax(pred[0])]
        
        if predicted_class == 'O':
            p_c = 'Wet Waste'
        else:
            p_c = 'Dry Waste'
        
        # Display the prediction
        st.write(f"Prediction: {p_c}")
        
    except Exception as e:
        st.error(f"Error: {e}")
