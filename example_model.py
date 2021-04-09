import random
import cv2
import numpy as np
from PIL import Image
from detect import detect
from detectreal import detectreal
class ExampleModel():
    def __init__(self):
        self.model = detect()

    # Generate an image based on some text.
    def run_on_input(self,input):
        #return np.array([1,2,3,4])
        return detectreal(self.model,input)

'''
        # This is an example of how you could use some input from
        # @runway.setup(), like options['truncation'], later inside a
        # function called by @runway.command().
        text = 'red'
        img = np.asarray(input['photo'])
        blured = cv2.blur(img,(5,5))
        print(blured)
        
        h, w = img.shape[:2]
        blured = cv2.blur(img,(5,5))    
        mask = np.zeros((h+2, w+2), np.uint8)  
        cv2.floodFill(blured, mask, (w-1,h-1), (255,255,255), (2,2,2),(3,3,3),8)
        cv2.floodFill(blured, mask, (0,0), (255,255,255), (2,2,2),(3,3,3),8)
        cv2.floodFill(blured, mask, (0,h-1), (255,255,255), (2,2,2),(3,3,3),8)
        cv2.floodFill(blured, mask, (w-1,0), (255,255,255), (2,2,2),(3,3,3),8)
        gray = cv2.cvtColor(blured,cv2.COLOR_BGR2GRAY) 
        #returnimg = Image.fromarray(np.uint8(gray))
        return np.array(gray)
'''