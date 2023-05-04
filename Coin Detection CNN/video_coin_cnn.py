import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import imutils

# Load the trained CNN model
model = tf.keras.models.load_model('coin_cnn_model.h5')

# Define a dictionary with the coin values
# coin_values = {1: 1, 2: 2, 5: 5, 10: 10}
coin_values = {0: 1, 1: 2, 2: 5, 3: 10}


# Define the lower and upper boundaries of the color range for each coin
coin_colors = {
    1: ([0, 0, 0], [180, 255, 80]),  # Black (1 rupee)
    2: ([0, 0, 80], [180, 50, 255]), # Red (2 rupees)
    5: ([0, 80, 80], [180, 255, 255]), # Golden (5 rupees)
    10: ([20, 80, 80], [35, 255, 255]) # Silver (10 rupees)
}

# Create a function to detect coins in each frame
def detect_coins(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect the coins in the frame
    coins = dict()
    for value, color in coin_colors.items():
        # Create a mask for the coin color range
        lower_bound, upper_bound = np.array(color[0]), np.array(color[1])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # Find the contours of the coins in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Draw a bounding box around each detected coin
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                coin = frame[y:y+h, x:x+w]
                coin = cv2.resize(coin, (180, 180))
                coin = np.expand_dims(coin, axis=0)
                coin = coin / 255.0
                pred = model.predict(coin)
                label = np.argmax(pred)
                print(value)
                print(label)
                print(coin_values[label])
                coins[value] = coins.get(value, 0) + coin_values[label]

                # Draw a bounding box around the coin
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, str(coin_values[label]) + ' rupees', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return coins, frame


# Capture the video feed from the camera
cap = cv2.VideoCapture(0)

# Loop through the frames in the video feed
while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    # Detect the coins in the frame and calculate the total amount
    coins, frame = detect_coins(frame)
    total_amount = sum(coins.values())

    # Display the total amount on the frame
    cv2.putText(frame, 'Total amount: ' + str(total_amount) + ' rupees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
    
    # show the frame on the screen
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()