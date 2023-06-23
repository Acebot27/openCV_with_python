import cv2
import numpy as np
import matplotlib.pyplot as plt

########################################################################################################################
# read image using cv2
img = cv2.imread('data_to_test/watch.jpg', cv2.IMREAD_GRAYSCALE)

# display image using cv2
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# display image using matplotlib
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.plot([200, 300, 400], [100, 200, 300], 'c', linewidth=5)
plt.show()

# save image
cv2.imwrite('data_to_test/watchgray.png', img)
del img                                     # delete image from memory

########################################################################################################################
# read video using cv2
# cap = cv2.VideoCapture('data_to_test/video_for_test.mp4')
cap = cv2.VideoCapture(0)  # for webcam
# cap = cv2.VideoCapture(1)  # for second webcam

# use codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('data_to_test/output.avi', fourcc, 20.0, (640,480))

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    out.write(frame)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):       # press 'q' to quit the video stream
        break

cap.release()
out.release()
cv2.destroyAllWindows()
del cap, out, fourcc, ret, frame, gray

########################################################################################################################
# drawing and writing on image using cv2
img = cv2.imread('data_to_test/watch.jpg', cv2.IMREAD_COLOR)
# drawing
cv2.line(img, (0, 0), (200, 300), (255, 255, 255), 20)
cv2.rectangle(img, (50, 25), (100, 250), (0, 0, 255), 15)
cv2.circle(img, (100, 63), 63, (0, 255, 0), -1)             # -1 filled the circle instead of line thickness
pts = np.array([[10, 50], [200, 250], [70, 250], [50, 10]], np.int32)
# pts = pts.reshape((-1, 1, 2))
cv2.polylines(img, [pts], True, (0, 255, 255), 3)

# writing
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV Tuts!', (10, 60), font, 1, (200, 255, 155), 3, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
del img                                     # delete image from memory

########################################################################################################################
# image operations
img = cv2.imread('data_to_test/watch.jpg', cv2.IMREAD_COLOR)
# creating a square box on image
img[100:150, 100:150] = [255, 255, 255]

# ROI: Region of Image
# Selecting and copying a region of image
watch_face = img[37:111, 107:194]
# place the ROI on top-left corner of the actual image
img[0:74, 0:87] = watch_face

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# certain characteristics of image
print('image shape:', img.shape)            # returns a tuple of number of rows, columns and channels
print('image size:', img.size)              # returns total number of pixels accessed
print('image datatype:', img.dtype)         # returns image datatype
del img                                     # delete image from memory
# Note: Other operation to merge two images are by using cv2.add() and cv2.addWeighted() methods
#       cv2.add() method adds each corresponding pixel of two images so ends up with brighter image (white region).
#       cv2.addWeighted() method applies following equation on the image: dst = src1*alpha + src2*beta + gamma;
#       where, alpha & beta are weight parameters and gamma is a scalar added to each sum. src1, src2 are input images.

########################################################################################################################
# image thresholding
img = cv2.imread('data_to_test/bookpage.jpg')

# grayscaling
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# simple thresholding
retval, threshold = cv2.threshold(src=img, thresh=12, maxval=255, type=cv2.THRESH_BINARY)
# maxval is the value that is assigned if the pixel value is more than the threshold value
gray_retval, gray_threshold = cv2.threshold(src=grayscaled, thresh=12, maxval=255, type=cv2.THRESH_BINARY)

# adaptive thresholding
# adaptive thresholding is used when image has different lighting conditions in different areas.
# adaptive thresholding calculates the threshold for a small region of image so we get different thresholds for
# different regions of same image and it gives us better results for images with varying illumination.
# adaptive thresholding takes three arguments: src, maxval and adaptive method
# adaptive method decides how thresholding value is calculated
# cv2.ADAPTIVE_THRESH_MEAN_C: threshold value is the mean of neighbourhood area
# cv2.ADAPTIVE_THRESH_GAUSSIAN_C: threshold value is the weighted sum of neighbourhood values where weights are a
#                                   gaussian window
# blocksize: decides the size of neighbourhood area
# C: is just a constant which is subtracted from the mean or weighted mean calculated
adaptive_threshold = cv2.adaptiveThreshold(src=grayscaled, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           thresholdType=cv2.THRESH_BINARY, blockSize=115, C=1)

# Otsu's thresholding
# Otsu's thresholding is used to automatically calculate threshold value of an image from its histogram
# for this, we use cv2.threshold() method but pass an extra flag, cv2.THRESH_OTSU
# for threshold value, simply pass 0
retval2, otsu_threshold = cv2.threshold(src=grayscaled, thresh=0, maxval=255, type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('original', img)
cv2.imshow('threshold', threshold)
cv2.imshow('gray_threshold', gray_threshold)
cv2.imshow('adaptive_threshold', adaptive_threshold)
cv2.imshow('otsu_threshold', otsu_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()
del img, grayscaled, threshold, gray_threshold, adaptive_threshold, otsu_threshold

########################################################################################################################
# color filtering
# color filtering is used to filter out an object based on color
# for this, we need to convert image from BGR to HSV
# HSV: Hue, Saturation, Value
# Hue: is the color
# Saturation: is the amount of color
# Value: is brightness of color
# in BGR, it is difficult to represent a color but in HSV, it is easy to represent a color
# in HSV, Hue range is [0, 179], Saturation range is [0, 255] and Value range is [0, 255]
# in OpenCV, Hue range is [0, 179], Saturation range is [0, 255] and Value range is [0, 255]
# in Matplotlib, Hue range is [0, 360], Saturation range is [0, 1] and Value range is [0, 1]
# in OpenCV, we use cv2.cvtColor() method to convert image from BGR to HSV
# in Matplotlib, we use cv2.COLOR_BGR2HSV and cv2.COLOR_HSV2BGR methods to convert image from BGR to HSV and vice versa
