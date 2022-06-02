import cv2
import numpy as np

# Load the image
img = cv2.imread("C:\\Users\\simon\\Documents\\GitHub\\photos_for_test_reg\\checkerboard_test\\test1.jpg")
print('Original Dimensions : ',img.shape)
 
scale_percent = 150 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
  
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
cv2.imshow("fnl", img)

# Color-segmentation to get binary mask
lwr = np.array([0, 0, 143])
upr = np.array([179, 61, 252])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
msk = cv2.inRange(hsv, lwr, upr)

# Extract chess-board
krn = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 30))
dlt = cv2.dilate(msk, krn, iterations=5)
res = 255 - cv2.bitwise_and(dlt, msk)
cv2.imshow("fnl", res)
# Displaying chess-board features
res = np.uint8(res)
ret, corners = cv2.findChessboardCorners(res, (46,41),
                                         flags=cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
if ret:
    print(corners)
    fnl = cv2.drawChessboardCorners(img, (7, 7), corners, ret)
    cv2.imshow("fnl", fnl)
    cv2.waitKey(0)
else:
    print("No Checkerboard Found")
cv2.waitKey(0)