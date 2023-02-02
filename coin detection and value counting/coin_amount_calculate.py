"""
Application of coin recognition

Task: Calculate the total value of coins in the picture

"""

import cv2
import numpy as np

image_name = '2019_1.jpg'

def detect_coins(filename):
    """
    Detect coin using Hough transform
    :param filename: name of the image to be scanned
    :return: the coordinates of circle centre and radius of coin
    """
    
    coins = cv2.imread('input_image/' + filename, 1)

    gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(gray, (15, 15), 0)
    detected_circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 28, param1 = 120, param2 = 35, minRadius = 15, maxRadius = 90)

    for circle in detected_circles[0]:
        x_coor, y_coor, detected_radius = circle
        coins_detected = cv2.circle(coins, (int(x_coor), int(y_coor)), int(detected_radius), (0, 255, 250), 3)

    cv2.imwrite('output_image/coin_amount/' + filename, coins_detected)

    return detected_circles

def calculate_amount(filename):
    """
    Calculates the value of coins in the image and displays it over the image
    :param filename: name of the image to be scanned
    :return: -
    """
    ## 2007 hasta mudra and common circulation series
    # coins_dim = {
    #     '₹1': {'value': 1, 'radius': 25, 'ratio': 1.086, 'count': 0,},
    #     '₹2': {'value': 2, 'radius': 27, 'ratio': 1.179, 'count': 0,},
    #     '₹5': {'value': 5, 'radius': 23, 'ratio': 1, 'count': 0,},
    #     #'₹10': {'value': 10, 'radius': 27, 'ratio': 1, 'count': 0,},
    #     #'50 Ps.': {'value': 0.5, 'radius': 22, 'ratio': 0.814, 'count': 0, },
    # }

    # ## 2011 rupee symbol series
    # coins_dim = {
    #     '₹1': {'value': 1, 'radius': 21.93, 'ratio': 1, 'count': 0,},
    #     '₹2': {'value': 2, 'radius': 25, 'ratio': 1.139, 'count': 0,},
    #     '₹5': {'value': 5, 'radius': 23, 'ratio': 1.048, 'count': 0,},
    #     '₹10': {'value': 10, 'radius': 27, 'ratio': 1.231, 'count': 0,},
    #     #'50 Ps.': {'value': 0.5, 'radius': 19, 'ratio': 0.76, 'count': 0, },
    # }

    ## 2019 grain series
    coins_dim = {
        '₹1': {'value': 1, 'radius': 21.93, 'ratio': 1, 'count': 0,},
        '₹2': {'value': 2, 'radius': 23, 'ratio': 1.048, 'count': 0,},
        '₹5': {'value': 5, 'radius': 25, 'ratio': 1.139, 'count': 0,},
        '₹10': {'value': 10, 'radius': 27, 'ratio': 1.231, 'count': 0,},
        # '₹20': {'value': 20, 'radius': 27, 'ratio': 1.231, 'count': 0, },
    }
    
    detected_circles = detect_coins(filename)
    radius = []
    coordinates = []
    
    for circle in detected_circles[0]:
        x_coor, y_coor, detected_radius = circle
        radius.append(int(detected_radius))
        coordinates.append([int(x_coor), int(y_coor)])

    # print(radius)
    smallest = min(radius)
    tolerance = 0.042
    total_amount = 0
    
    coins_circled = cv2.imread('output_image/coin_amount/' + filename, 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for coin in detected_circles[0]:
        ratio_to_check = coin[2]/ smallest
        coor_x = int(coin[0])
        coor_y = int(coin[1])
        
        for denomination in coins_dim:
            value = coins_dim[denomination]['value']
            if abs(ratio_to_check - coins_dim[denomination]['ratio']) <= tolerance:
                coins_dim[denomination]['count'] += 1
                total_amount += coins_dim[denomination]['value']
                cv2.putText(coins_circled, str(value), (coor_x, coor_y), font, 1, (0, 0, 0), 4)
        
    print(f"The total amount is ₹{total_amount}")
    for denomination in coins_dim:
        pieces = coins_dim[denomination]['count']
        print(f"{denomination} = {pieces}x")


    cv2.imwrite('output_image/coin_amount/' + filename[:-4] + '_value.jpg', coins_circled)

if __name__ == "__main__":
    calculate_amount(image_name)