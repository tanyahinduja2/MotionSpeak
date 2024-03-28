import cv2 as cv
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

capture = cv.VideoCapture(0)
detector = HandDetector(maxHands=2)
offset = 20
imgSize = 300
folder = "Data/A"
count = 0

while True:
    success, img = capture.read()
    hands, img = detector.findHands(img)
    if hands:
        x_min = min(hand['bbox'][0] for hand in hands)
        y_min = min(hand['bbox'][1] for hand in hands)
        x_max = max(hand['bbox'][0] + hand['bbox'][2] for hand in hands)
        y_max = max(hand['bbox'][1] + hand['bbox'][3] for hand in hands)

        imgCrop = img[y_min - offset:y_max + offset, x_min - offset:x_max + offset]

        h, w, _ = imgCrop.shape
        aspect_ratio = w / h
        if aspect_ratio > 1:
            calcWidth = imgSize
            calcHeight = int(calcWidth / aspect_ratio)
        else:
            calcHeight = imgSize
            calcWidth = int(calcHeight * aspect_ratio)

        imgResize = cv.resize(imgCrop, (calcWidth, calcHeight))

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        x_offset = (imgSize - calcWidth) // 2
        y_offset = (imgSize - calcHeight) // 2
        imgWhite[y_offset:y_offset + calcHeight, x_offset:x_offset + calcWidth] = imgResize

        cv.imshow("ImageWhite", imgWhite)

    cv.imshow("Image", img)
    key = cv.waitKey(1)
    if key == ord("s") :
        count += 1
        cv.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(count)