#Working code 
import cv2
import numpy as np
from keras.models import load_model

# Load your pre-trained model
model = load_model('waste.h5')
class_names=['O', 'R']
def preprocess_image(img):
    # Resize the image to match the input size of the model
    img = cv2.resize(img, (242, 208))
    # Normalize the pixel values if required
    # img = img / 255.0
    return img

def predict_image(model, img):
    # Expand dimensions to match the model's expected input shape
    img = np.expand_dims(img, axis=0)
    # Make predictions
    prediction = model.predict(img)
    return prediction

# Open the camera (0 is usually the default camera)
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the frame
    cv2.imshow('Camera', frame)

    # Check if the user pressed the 'q' key to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Check if the user pressed the 'c' key to capture an image
    elif cv2.waitKey(1) & 0xFF == ord('c'):
        # Preprocess the captured frame
        processed_frame = preprocess_image(frame)

        # Make predictions
        prediction = predict_image(model, processed_frame)
        print(prediction)
        predicted_class = class_names[np.argmax(prediction[0])]
        if(predicted_class == 'O'):
            print('Wet Waste')
        else:
            print('Dry Waste')

        # You can add more logic here to handle the prediction result

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
