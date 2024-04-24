from flask import Flask, render_template, request
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from joblib import load

app = Flask(__name__)

# Load the CNN model
cnn_model = load_model('cnn_model.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the RandomForest model using joblib
rf_model = load('rf_model.pkl')

def extract_roi(image):
    try:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the eye cascade classifier
        eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")

        # Detect eyes in the grayscale image
        eyes = eyesCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))

        if len(eyes) != 0:
            for (x, y, w, h) in eyes:
                eye = image[y:y + h, x:x + w]
                ROI_reduced = reduce_size_of_image(eye)  # Reduce size of the ROI
                cv2.imwrite("./images/ROI_Reduced.jpg", ROI_reduced)  # Save the reduced ROI image
                return ROI_reduced.astype(np.uint8)  # Ensure ROI is of type uint8
        else:
            return None
    except Exception as e:
        print(f"Error in extract_roi function: {e}")
        return None

def reduce_size_of_image(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 1.6)   ## Can be made ~25% 
    width_left = int(width / 6)
    width_right = int(width / 6)
    img = img[eyebrow_h:height, width_left:(width - width_right)]  # Crop the ROI
    return img

@app.route('/', methods=['GET', 'POST'])
def detect():
    classification = ""  # Initialize classification variable

    if request.method == 'POST':
        try:
            if 'imagefile' not in request.files:
                return "Error: No file uploaded"
            
            imagefile = request.files['imagefile']
            image_path = "./images/" + imagefile.filename
            imagefile.save(image_path)

            # Load and preprocess the image
            user_img = cv2.imread(image_path)
            user_img_gray = cv2.cvtColor(user_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            user_img_resized = cv2.resize(user_img_gray, (100, 100))  # Resize image to match model input shape
            user_img_normalized = user_img_resized.astype('float32') / 255.0  # Normalize the image

            # Convert grayscale image to three-channel image
            user_img_rgb = cv2.cvtColor(user_img_normalized, cv2.COLOR_GRAY2RGB)

            # Convert RGB image to LAB color space
            user_img_lab = cv2.cvtColor(user_img_rgb, cv2.COLOR_RGB2LAB)

            # Expand dimensions to add batch dimension
            user_img_lab = np.expand_dims(user_img_lab, axis=0)

            # Extract ROI
            roi_extracted = extract_roi(user_img)
            if roi_extracted is not None:

                # Perform prediction with the CNN model
                ip_feat = cnn_model.predict(user_img_lab)

                # Perform prediction with the RandomForest model
                prd = rf_model.predict(ip_feat)

                # Determine classification
                classification = "You are anemic" if prd[0] == 0 else "You are non-anemic"
            else:
                classification = "Error: No eyes detected in the image"
        except Exception as e:
            classification = f"Error: {str(e)}"
            print(classification)

    return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=False)
