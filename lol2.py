import numpy as np
import cv2
from mss import mss
from PIL import Image



mon = {'top': 40, 'left': 0, 'width': 800, 'height': 580}

sct = mss()

while 1:
    sct.get_pixels(mon)
    img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
    cv2.imshow('GrabLane', np.array(img))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break