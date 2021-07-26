import uuid

import cv2
import numpy as np
import tensorflow as tf

# top, bottom, left (flipped), right (flipped)
REGION_OF_INTEREST_COORDINATES = (10, 600, 600, 10)

# CAPTURE = cv2.VideoCapture(3)
CAPTURE = cv2.VideoCapture(0)

FINGERS = 6
COUNT = 600

if FINGERS == 6:
    COUNT = COUNT * 2

frame_index = 0
while True:
    _, frame = CAPTURE.read()
    frame = cv2.flip(frame, 1)

    region_of_interest = frame[
        REGION_OF_INTEREST_COORDINATES[0] : REGION_OF_INTEREST_COORDINATES[1],
        REGION_OF_INTEREST_COORDINATES[3] : REGION_OF_INTEREST_COORDINATES[2],
    ]
    region_of_interest = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
    region_of_interest = cv2.GaussianBlur(region_of_interest, (7, 7), 0)

    kernel = np.ones((5,5),np.uint8)
    region_of_interest = cv2.morphologyEx(region_of_interest, cv2.MORPH_CLOSE, kernel)
    
    # black background, white hand
    _, region_of_interest = cv2.threshold(
        region_of_interest, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    
    cv2.imshow("FINGER DETECTION", region_of_interest)

    hand = cv2.resize(region_of_interest, (128, 128))

    # Press q to exit the application
    ch = cv2.waitKey(1) & 0xFF
    if ch == ord("q"):
        break
    elif ch == ord("s"):
        if FINGERS == 6:
            cv2.imwrite(f"data/fingers-binary-thresh/test/{FINGERS}/{uuid.uuid4()}_{FINGERS}N.jpg", hand)
            
        else:
            cv2.imwrite(f"data/fingers-binary-thresh/test/{FINGERS}/{uuid.uuid4()}_{FINGERS}R.jpg", hand)
            hand = cv2.flip(hand, 1)
            cv2.imwrite(f"data/fingers-binary-thresh/test/{FINGERS}/{uuid.uuid4()}_{FINGERS}L.jpg", hand)

        COUNT = COUNT - 1
        print(COUNT)
        
        if (COUNT <= 0):
            break


CAPTURE.release()
cv2.destroyAllWindows()
