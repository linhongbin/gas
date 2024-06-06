import cv2
import numpy as np
image = cv2.imread("./gym_ras/asset/surgery_background.jpeg")

cx = 0.5
cy = 0.5
length = 0.9
width = 0.9
angle =0.25



print(image.shape)
print(np.int(cx*image.shape[0]))
rot_rectangle = ((np.int(cy*image.shape[1]),np.int(cx*image.shape[0]), ), 
                    (np.int(length*image.shape[1]), np.int(width*image.shape[0])), np.int(angle*180))
box = cv2.boxPoints(rot_rectangle) 
print(box)
box = np.int0(box) #Convert into integer values
print(box)
rectangle = cv2.drawContours(image,[box],contourIdx=0,color=(0,0,255),thickness=cv2.FILLED)

cv2.imshow("Rotated Rectangle",rectangle)

cv2.waitKey(0)
cv2.destroyAllWindows()
