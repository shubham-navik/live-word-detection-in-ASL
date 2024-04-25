from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import pickle
import mediapipe as mp

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes by passing the Flask app to CORS constructor

# Load the pre-trained model and labels dictionary
model_dict = pickle.load(open(r'C:\Users\shubham\OneDrive\Documents\Desktop\signlanguage\experiment\models\TrainedModel\model.p', 'rb'))
model = model_dict['model']
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 
    6: 'G', 7: 'H', 8: 'I', 9: 'K', 10: 'L', 11: 'M', 
    12: 'N', 13: 'O', 14: 'P', 15: 'Q', 16: 'R', 17: 'S', 
    18: 'T', 19: 'U', 20: 'V', 21: 'W', 22: 'X', 23: 'Y'
}



# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Preprocess function for image data
def preprocess_image(image_data):
    try:
        # Read the image using OpenCV
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
        return image
    except Exception as e:
        print("Error preprocessing image:", e)
        return None

# Process image and predict gesture
def process_image(image):
    try:
        data_aux = []
        x_ = []
        y_ = []

        # Get frame dimensions
        H, W, _ = image.shape

        # Convert the image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process hand landmarks
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Predict gesture using the model
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Return the predicted character
            return predicted_character
    except Exception as e:
        print("Error processing image:", e)
        return None

# Define a route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        if not image_file:
            return jsonify({'error': 'No image provided'})

        # Read the image file as bytes
        image_data = image_file.read()

        # Preprocess the image data
        image = preprocess_image(image_data)
        if image is None:
            return jsonify({'error': 'Failed to preprocess image'})

        # Process the image to predict gesture
        predicted_character = process_image(image)
        if predicted_character is None:
            return jsonify({'space': ' '})

        # Return the predicted character in the response
        return jsonify({'success': True, 'prediction': predicted_character})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app
