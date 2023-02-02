"""
coin detection using opencv and python
task: draw a circle around each coin in an image
method:
- find the edge using Gaussian and Canny
- try to fit a circle to the edges by comparing circles of increasing size. Once passed threshold,
  assume that is the radius of coin and save the coordinates of the center
- draw the circles to the original image
"""

import cv2
import numpy as np
import math

filename = 'coins3.jpg'
# filename = 'coindetection.jpg'

coins = cv2.imread('input_image/'+filename, 1)

# defining minimal and maximal radius, specified to the image
min_r = 12
max_r = 85

# detect edges
def edge_detect_coins():
    """
    detect the edge of the coins
    :return: image of coin edges
    """
    coins_height, coins_width, coins_Channel = coins.shape

    # optimisation by decreasing the size of image, resulting in 4x faster run time
    coins_resized = cv2.resize(coins, (int(coins_width/2), int(coins_height/2)))

    # blur to optimise edge detection
    coins_blurred = cv2.GaussianBlur(coins_resized, (5,5), cv2.BORDER_DEFAULT)

    # use Canny to find the edge
    coins_edge = cv2.Canny(coins_blurred, 127, 255)

    cv2.imwrite('output_image/' + filename[:-4]+ '_blurred.jpg', coins_blurred)
    cv2.imwrite('output_image/' + filename[:-4]+ 'coins_edge.jpg', coins_edge)

    return coins_edge

# detect centres
def coin_centre_detect():
    """
    aim is to find the edges, find the radius of the coin and save the co-ordinates of the centre
    :return: list of co-ordinates of centre with radius of coins
    """

    # image with edges of coin detected
    coins_edge = edge_detect_coins()

    # obtain the image size
    max_height, max_width = coins_edge.shape

    edge_threshold = 0.35 # how many pixels need to pass to be considered a coin edge
    intensity_threshold = 255*0.123 # the min value of a pixel intensity to be considered edge
    next_circle_step = 1 # the amount of pixels to move to start comparing again
    coin_detection = []


    for radius in range(min_r, max_r):
        img_circle = np.zeros((radius*2, radius*2, 1), np.uint8)
        circle = cv2.circle(img_circle, (radius, radius), radius, 255)

        circumference = 2 * math.pi * radius

        circle_pixels = []

        for y in range(len(circle)):
            for x in range(len(circle[y])):
                if circle[x][y] == 255:
                    circle_pixels.append((x, y))

        print(('radius', radius))

        # move circle through image
        for start_y in range(0, max_height-2*radius, next_circle_step):
            for start_x in range(0, max_width-2*radius, next_circle_step):
                count = 0

                # cycle through the co-ordinates of circle
                for(x,y) in circle_pixels:
                    image_y = start_y + y
                    image_x = start_x + x

                    if coins_edge[image_y][image_x] >= intensity_threshold:
                        count += 1

                if count > 50:
                    percentage = round(count / circumference*100, 2)
                    coor_x = start_x + radius
                    coor_y = start_y + radius
                    print(("candidate", coor_x, coor_y, radius, percentage))

                if (count / circumference) > edge_threshold:
                    coor_x = start_x + radius
                    coor_y = start_y + radius
                    coin_detection.append((coor_x, coor_y, radius)) # centre
                    print(("-"*18, start_x + radius, start_y + radius, radius))

    return coin_detection

# draw circles
def circle_coins():

    coins_circled = coin_centre_detect()
    coins_copy = coins.copy()

    for detect_circle in coins_circled:
        x_coor, y_coor, detected_radius = detect_circle
        coins_detected = cv2.circle(coins_copy, (x_coor, y_coor), detected_radius*2, (55, 55,55), 2)

    cv2.imwrite('output_image/coin_detection/'+ filename[:-4]+ '_detected.jpg', coins_detected)

def hough_circle_detection():
    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(gray, 11)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 28, param1 = 180, param2 = 30, minRadius = min_r*2, maxRadius = max_r*2)

    coins_copy = coins.copy()

    for detected_circle in circles[0]:
        x_coor, y_coor, detected_radius = detected_circle

        x_coor = x_coor.astype(int)
        y_coor = y_coor.astype(int)
        detected_radius = detected_radius.astype(int)

        coins_detected_h = cv2.circle(coins_copy, (x_coor, y_coor), detected_radius, (55, 55, 55), 2)

    cv2.imwrite('output_image/coin_detection/' + filename[:-4]+ '_hough.jpg', coins_detected_h)


# def compare_circle_detection():
# circle_coins()
hough_circle_detection()


