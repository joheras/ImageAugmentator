import cv2
import numpy as np

# A box is a tuple (category,(x,y,w,h)) where category is the category of the
# object, x and y represent the top-left corner, w is the width and h is the height.
def detectBox(image,box,function):
    mask = np.zeros(image.shape[:2], dtype="uint8")
    (category,(x,y,w,h)) = box
    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    newmask = apply(function,[mask])
    (cnts, _) = cv2.findContours(newmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if(len(cnts)==0):
        return None
    return (category,cv2.boundingRect(cnts[0]))


# Boxes is a list of boxes with the following format: (category,(x,y,w,h))
def detectBoxes(image,boxes,function):
    return [detectBox(image,box,function) for box in boxes]

# Example
# def flip1(image):
#     return cv2.flip(image,1)
# detectBox(image,('dog',(0,0,20,20)),flip1)






