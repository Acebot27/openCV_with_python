import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    #############################################################################################
    # color filtering
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([30, 150, 50])
    upper_red = np.array([255, 255, 180])

    mask = cv2.inRange(src=hsv, lowerb=lower_red, upperb=upper_red)
    res = cv2.bitwise_and(src1=frame, src2=frame, mask=mask)
    #############################################################################################
    # smoothing method
    kernel = np.ones((5, 5), np.float32) / 25   # 5x5 matrix with all ones and divide by 25 to get average of 25
    smoothed = cv2.filter2D(src=res, ddepth=-1, kernel=kernel)
    # ddepth = -1 means the output image will have the same depth as the source
    # kernel is a window which is used to calculate the pixel value by adding all the pixels in the image under the
    # kernel and dividing the sum by the number of pixels in the kernel
    # kernel is used to apply effects like blurring, sharpening, edge detection, and more

    # blurring method
    blur = cv2.GaussianBlur(src=res, ksize=(5, 5), sigmaX=0)
    # ksize is the kernel size
    # sigmaX is the standard deviation in X direction

    # median method
    median = cv2.medianBlur(src=res, ksize=5)
    # ksize is the kernel size

    # bilateral method
    bilateral = cv2.bilateralFilter(src=res, d=9, sigmaColor=75, sigmaSpace=75)
    # d is the diameter of each pixel neighborhood that is used during filtering
    # sigmaColor is the standard deviation in the color space
    # sigmaSpace is the standard deviation in the coordinate space
    #############################################################################################

    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    cv2.imshow('res', res)
    cv2.imshow('smoothed', smoothed)
    cv2.imshow('bilateral', bilateral)
    cv2.imshow('blur', blur)
    cv2.imshow('median', median)

    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
del cap, frame, _, hsv, lower_red, upper_red, mask, res, kernel, smoothed, blur, median, bilateral, k

# #############################################################################################