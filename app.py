from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
from keras.models import load_model
import base64
import cv2

#model = pickle.load(open('file.pkl','rb'))
model = load_model('waste.h5')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index1.html')

'''
@app.route('/submit_images')
def submit_images():
    for i in range(10):
        encoded_image_str = request.POST.get(f"image{i}")
        encoded_image_bytes = encoded_image_str.encode("utf-8")
        decoded_image_bytes = base64.b64decode(encoded_image_bytes)

        # Save decoded image to file
        with open(f"opencv{i}.png", "wb") as f:
            f.write(decoded_image_bytes)
'''

@app.route('/predict', methods=['POST'])
def result():
    print('0st if')
    file = request.files['image']
    file.save('static/file.jpeg')
    print(file)
    # Check if the post request has the file part
    if 'image' not in request.files:
        print('1st if')
        return render_template('result.html', error='No file part')


    # If the user does not select a file, submit an empty part without filename
    if file.filename == '':
        print('2st if')
        return render_template('result.html', error='No selected file')

    # If the file exists and is allowed, read the content
    if file:
        try:
            class_names=['O', 'R']
            print('3st if')
            # Read the image file
            img = Image.open(file)

            # Perform any necessary preprocessing (resize, normalize, etc.)
            # ...
            img = img.resize((242, 208))
            # Convert the image to a numpy array
            img_array = np.array(img)
            #image_data = base64.b64encode(img_array.tobytes()).decode('utf-8')

            #image_data_uri = f"data:image/jpeg;base64,{image_data}"
            # Make predictions using the model
            #pred = model.predict(img_array)
            pred = model.predict(np.expand_dims(img_array, axis=0))
            print('prediction', pred)
            predicted_class = class_names[np.argmax(pred[0])]
            print(predicted_class)
            if(predicted_class == 'O'):
                p_c= 'Wet Waste'
            else:
                p_c= 'Dry Waste'
            #plt.imshow(img)
            #return pred
            return render_template('result.html', data=pred, data1=p_c, image_path='static/file.jpeg')

        except Exception as e:
            return render_template('result.html', error=str(e))



if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)