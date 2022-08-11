# _*_ coding: UTF-8 _*_
#1.Referenced library (module)
import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)  #  2.Open the camera The built-in camera index is 0, 0 represents the camera number, and if there is only one, the default is 0
'''
cap = cv2.VideoCapture(0)The parameter 0 means that the default is the built-in first camera of the notebook
If you need to read an existing video image, change the parameter to the path where the image is located
例如：cap=cv2.VideoCapture(‘video.mp4’)，Of course, by collecting video images
'''

#3.Read photo Select location of gesture input
while (cap.isOpened()):     #Loop, check whether the initialization is successful, and return True if successful cap.isOpened() judges whether the video image object is successfully read, and returns True if the video image object is successfully read.
    ret, frame = cap.read()     # Read the image by frame, combined with the while loop can always read the video
    '''
    ret,frame = cap.read()Read the video image by frame. The return value ret is a boolean type. If it is read correctly, it will return True. If the reading fails or the end of the video image is read, it will return False. frame is the image of each frame
    '''

    frame = cv2.flip(frame, 1)            #Image rotation，cv2.flip(frame, 1)The first parameter indicates the video to be rotated, the second parameter indicates the direction of rotation, 0 indicates rotation around the x axis, a number greater than 0 indicates rotation around the y axis, and a negative number less than 0 indicates rotation around the x and y axes
    kernel = np.ones([2, 2], np.uint8)    #matrix assignment
    roi = frame[100:300, 100:300]       # Select a fixed position in the picture as gesture input

    cv2.rectangle(frame, (100, 100), (400, 400), (0, 0, 255), 0)  # Draw the gesture recognition box with a red line; picture, coordinates of the upper left point, coordinates of the lower right point, rgb color, width of the line

    # 4.skin color detection based on hsv
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)    #Convert an input image in one color space to another color space
    lower_skin = np.array([0, 28, 70], dtype=np.uint8)              #Find low threshold for color
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)              #Find high threshold for color

    # 5.Gaussian filtering
    mask = cv2.inRange(hsv, lower_skin, upper_skin)   #Use the cv2.inRange function to set the threshold to remove the background part
    '''
      mask = cv2.inRange(hsv, lower_red, upper_red) #lower20===>0,upper200==>0,
    The first parameter: hsv refers to the original image

    The second parameter: lower_red refers to the value lower than this lower_red in the image, and the image value becomes 0

   The third parameter: upper_red refers to the value higher than this upper_red in the image, and the image value becomes 0

   '''
    mask = cv2.dilate(mask, kernel, iterations=4)
    '''
    cv2.dilate(img, kernel, iteration)
    img – target image
    kernel – The kernel to operate on, defaults to a 3x3 matrix
    iterations – Corrosion times, the default is 1
    '''
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    '''

    The most commonly used parameters are: img = cv2.gaussian blur (src, (blur1, blur2), 0)
    Where src is the original image to be filtered, (blur1, blur2) is the size of Gaussian kernel, the selection of blur1 and blur2 is generally odd, and the values of blur1 and blur2 can be different. A parameter of 0 indicates that the standard deviation is 0.

    Detailed explanation of parameters in GaussianBlur () filter;
    cv2.GaussianBlur（ SRC，ksize，sigmaX [，DST [，sigmaY [，borderType ] ] ] ）
    Src–input image; Images can have any number of channels that are processed independently, but the depth should be CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.
    DST–Output image src with the same size and type as the image.
    K size–Gaussian kernel size. Ksize.width and ksize.height can be different, but they must all be positive and odd numbers. Or, they can be zero, and then calculate sigma* from.
    Max–Gaussian kernel standard deviation in x direction.
    Gaussian kernel standard deviation in SIG–Y direction; If sigmaY is zero, set it equal to sigmaX；;
    If both sigma are zero, calculate k size. height according to ksize.width and k size.
    The result of complete control, regardless of the semantic changes of all this in the future, is suggested to specify all ksize, sigmaX and sigmaY.
    Bordertype–pixel extrapolation method
    '''

    #6.find the outline
    contours, h = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Find contours of detected objects
    '''
    Note that the parameter of cv2.findContours () function is a binary image, that is, a black-and-white image (not a grayscale image), so the read image should be converted into a grayscale image first, and then into a binary image.
    The first parameter is to find the contour image;
    The second parameter indicates the retrieval mode of the outline. There are four types:
    2. cv2.RETR_EXTERNAL indicates that only the outer contour is detected.
    2. The contour detected by cv2.RETR_LIST does not establish a hierarchical relationship.
    2. cv2.RETR_CCOMP establishes two levels of contours, the upper layer is the outer boundary, and the inner layer is the boundary information of the inner hole. If there is another connected object in the inner hole, the boundary of this object is also at the top layer.
    2. cv2.RETR_TREE establishes the outline of a hierarchical tree structure.
    The third parameter, method, is the approximate method of contour.
    2. CVChain _ Proxy _ None stores all contour points, and the pixel position difference between two adjacent points is less than 1, that is, max(abs(x1-x2), abs(y2-y1))==1.
    2. Chain _ Proxy _ Simple compresses elements in horizontal, vertical and diagonal directions, and only keeps the coordinates of the end point of that direction. For example, a rectangular outline only needs 4 points to save the outline information.
    2. Chain _ approximate _ TC89 _ L1, CV _ chain _ approximate _ TC89 _ KCOS uses teh-Chinl chain approximation algorithm.
    '''

    #7.Define and find bumps
    cnt = max(contours, key=lambda x: cv2.contourArea(x))
    epsilon = 0.0005 * cv2.arcLength(cnt, True)    #Calculate contour perimeter cv.arcLength
    '''
    InputArray type curve, input vector, two-dimensional point (contour vertex), can be std::vector or Mat type.
    Closed of type bool, an identifier used to indicate whether the curve is closed, generally set to true.
    Note: The perimeter value of the calculated contour is calculated according to the actual length of the contour
    '''
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    '''
    cv2.approxPolyDP(contour,epsilon,True) uses the Douglas-Peucker algorithm
    The first parameter is the set of points for the contour.
    The meaning of the second parameter epsilon is as follows. The distance between the filtered line segment set and the newly generated line segment set is d. If d is less than epsilon, it is filtered out, otherwise it is retained.
    The third parameter indicates whether the newly generated contour is closed.
    The returned polygon is a series of points.
    '''
    hull = cv2.convexHull(cnt)  #Find Polygon Convex Hull
    '''
    Draw three outlines of the source image against a black background:
    Contours obtained by cv2.findContours
    Approximate polygon processing on the contours obtained by cv2.findContours cv2.approxPolyDP
    Do convex hull processing on the contour obtained by cv2.findContours cv2.convexHul
    '''
    areahull = cv2.contourArea(hull)  #Calculate the area of the contour of the hull image
    areacnt = cv2.contourArea(cnt)   #Calculate the area of the contour of the cnt image
    arearatio = ((areahull - areacnt) / areacnt) * 100

    # 8.Find the concave and convex points
    hull = cv2.convexHull(approx, returnPoints=False)
    defects = cv2.convexityDefects(approx, hull)
    # Define the initial value of the number of bump points to 0
    l = 0    #Define the initial value of the number of bump points to 0
    for i in range(defects.shape[0]):
        s, e, f, d, = defects[i, 0]
        start = tuple(approx[s][0])
        end = tuple(approx[e][0])
        far = tuple(approx[f][0])
        pt = (100, 100)

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)     # math.sqrt Returns the square root of different numbers.
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        s = (a + b + c) / 2
        ar = math.sqrt(s * (s - a) * (s - b) * (s - c))
        # Angle between fingers
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

        if angle <= 90 and d > 20:
            l += 1
            cv2.circle(roi, far, 3, [255, 0, 0], -1)
        cv2.line(roi, start, end, [0, 255, 0], 2)  # draw the envelope
    l += 1
    font = cv2.FONT_HERSHEY_SIMPLEX


    # 9.Conditional judgment, that is, what function you want to achieve after knowing the gesture, and adding the recognized pattern, which is equivalent to a library (defined according to the gesture characteristics).
    if l == 1:
        if areacnt < 2000:
            cv2.putText(frame, "put hand in the window", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            if arearatio < 12:
                cv2.putText(frame, '0', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            elif arearatio < 17.5:
                cv2.putText(frame, "1", (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                cv2.putText(frame, '1', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif l == 2:
        cv2.putText(frame, '2', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif l == 3:
        if arearatio < 27:
            cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        else:
            cv2.putText(frame, '3', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif l == 4:
        cv2.putText(frame, '4', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
    elif l == 5:
        cv2.putText(frame, '5', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
   #10.save display
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    k = cv2.waitKey(25) & 0xff
    #11.Keyboard Esc key to stop the program
    if k == 27:
        break
cv2.destroyAllWindows()
cap.release()