import numpy as np
import imutils
from joblib import  load
from skimage import feature
from robobo_video.robobo_video import RoboboVideo
import cv2, time
from Robobo import Robobo

#classification of blue signals
def test_blue(clf_blue, image):
    im_test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd_test_blue, h_test_blue = feature.hog(im_test_gray, orientations=7, pixels_per_cell=(8, 8),
                                            cells_per_block=(2, 2), transform_sqrt=False, block_norm="L1",
                                            visualize=True)

    hog = h_test_blue.reshape(64 * 64)
    predict = clf_blue.predict([hog])
    class_prob = clf_blue.predict_proba([hog])

    return predict, class_prob

#classification of red signals
def test_red(clf_red, image):
    im_test_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd_test_red, h_test_red = feature.hog(im_test_gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(4, 4),
                                          transform_sqrt=False, block_norm="L1", visualize=True)

    hog = h_test_red.reshape(64 * 64)
    predict = clf_red.predict([hog])
    class_prob = clf_red.predict_proba([hog])

    return predict, class_prob

#search and crop the image
def identify_blue(imag, clf_blue):
    label_list = list()
    cnts_list = list()
    mser_blue = cv2.MSER_create(8, 900, 2300)

    img = imag.copy()
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # convert the image to HSV format for color segmentation
    img_hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)

    # mask to extract blue
    lower_blue = np.array([90, 10, 10])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)

    blue_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

    # seperate out the channels
    r_channel = blue_mask[:, :, 2]
    g_channel = blue_mask[:, :, 1]
    b_channel = blue_mask[:, :, 0]

    # filter out
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)

    # create a blue gray space
    filtered_b = + 2.5 * filtered_b - 0.5 * filtered_g

    # Do MSER
    regions, _ = mser_blue.detectRegions(np.uint8(filtered_b))

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    blank = np.zeros_like(blue_mask)
    cv2.fillPoly(np.uint8(blank), hulls, (255, 0, 0))

    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(blank, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)

    _, b_thresh = cv2.threshold(opening[:, :, 0], 60, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(b_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_cnts = 3  # no frame we want to detect more than 3

    if not cnts == []:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        if len(cnts_sorted) > max_cnts:
            cnts_sorted = cnts_sorted[:3]

        for c in cnts_sorted:
            x, y, w, h = cv2.boundingRect(c)
            if x < 100:
                continue
            if h < 20:
                continue

            if y > 400:
                continue

            aspect_ratio_1 = w / h
            aspect_ratio_2 = h / w
            if aspect_ratio_1 <= 0.5 or aspect_ratio_1 > 1.2:
                continue
            if aspect_ratio_2 <= 0.5:
                continue

            hull = cv2.convexHull(c)

            cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)

            mask = np.zeros_like(imag)
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)  # Draw filled contour in mask
            cv2.rectangle(mask, (x, y), (int(x + w), int(y + h)), (255, 255, 255), -1)
            out = np.zeros_like(imag)  # Extract out the object and place into output image
            out[mask == 255] = imag[mask == 255]

            x_pixel, y_pixel, _ = np.where(mask == 255)
            (topx, topy) = (np.min(x_pixel), np.min(y_pixel))
            (botx, boty) = (np.max(x_pixel), np.max(y_pixel))


            out = imag[topx:botx + 1, topy:boty + 1]
            out_resize = cv2.resize(out, (64, 64), interpolation=cv2.INTER_CUBIC)

            # prediction
            predict, prob = test_blue(clf_blue, out_resize)
            print(np.max(prob))

            if np.max(prob) < 0.9:
                continue
            cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
            label = predict[0]
            if label == 100:
                continue

            cnts_list.append(c)
            label_list.append(label)
        return cnts_list, label_list
    else:
        return None, None

#search and crop the image
def identify_red(imag, clf_red):
    label_list = list()
    cnts_list = list()
    mser_red = cv2.MSER_create(8, 1000, 1800)

    img = imag.copy()
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # mask to extract red
    img_hsv = cv2.cvtColor(imag, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0,100,100])
    upper_red_1 = np.array([10, 255, 255])
    mask_1 = cv2.inRange(img_hsv, lower_red_1, upper_red_1)
    lower_red_2 = np.array([170, 70, 60])
    upper_red_2 = np.array([180, 255, 255])
    mask_2 = cv2.inRange(img_hsv, lower_red_2, upper_red_2)

    mask = cv2.bitwise_or(mask_1, mask_2)
    red_mask = cv2.bitwise_and(img_output, img_output, mask=mask)

    # separating channels
    r_channel = red_mask[:, :, 2]
    g_channel = red_mask[:, :, 1]
    b_channel = red_mask[:, :, 0]

    # filtering
    filtered_r = cv2.medianBlur(r_channel, 5)
    filtered_g = cv2.medianBlur(g_channel, 5)
    filtered_b = cv2.medianBlur(b_channel, 5)

    filtered_r =  2* filtered_r - .5 * filtered_b - .5 * filtered_g

    # MSER detection
    regions, _ = mser_red.detectRegions(np.uint8(filtered_r))

    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    blank = np.zeros_like(red_mask)
    cv2.fillPoly(np.uint8(blank), hulls, (0, 0, 255))  # fill a blank image with the detected hulls
    # perform some operations on the detected hulls from MSER
    kernel_1 = np.ones((3, 3), np.uint8)
    kernel_2 = np.ones((5, 5), np.uint8)

    erosion = cv2.erode(blank, kernel_1, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_2)

    _, r_thresh = cv2.threshold(opening[:, :, 2], 20, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(r_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    max_cnts = 3  # no frame we want to detect more than 3
    if not cnts == []:
        cnts_sorted = sorted(cnts, key=cv2.contourArea, reverse=True)
        if len(cnts_sorted) > max_cnts:
            cnts_sorted = cnts_sorted[:3]

        for c in cnts_sorted:
            x, y, w, h = cv2.boundingRect(c)
            if x < 100:
                continue
            if h < 20:
                continue

            if y > 400:
                continue

            aspect_ratio_1 = w / h
            aspect_ratio_2 = h / w
            if aspect_ratio_1 <= 0.3 or aspect_ratio_1 > 1.2:
                continue
            if aspect_ratio_2 <= 0.3:
                continue

            hull = cv2.convexHull(c)
            mask = np.zeros_like(imag)
            cv2.drawContours(mask, [c], -1, (255, 255, 255), -1)  # Draw filled contour in mask
            cv2.rectangle(mask, (x, y), (int(x + w), int(y + h)), (255, 255, 255), -1)

            out = np.zeros_like(imag)  # Extract out the object and place into output image
            out[mask == 255] = imag[mask == 255]

            x_pixel, y_pixel, _ = np.where(mask == 255)
            (topx, topy) = (np.min(x_pixel), np.min(y_pixel))
            (botx, boty) = (np.max(x_pixel), np.max(y_pixel))

            out = imag[topx:botx + 1, topy:boty + 1]

            out_resize = cv2.resize(out, (64, 64), interpolation=cv2.INTER_CUBIC)

            #prediction
            predict, prob = test_red(clf_red, out_resize)
            print(np.max(prob))

            if np.max(prob) < 0.9:
                continue
            cv2.rectangle(imag, (x, y), (int(x + w), int(y + h)), (0, 255, 0), 2)
            label = predict[0]
            if label == 100:
                continue
            cnts_list.append(c)
            label_list.append(label)
        return cnts_list, label_list
    else:
        return None, None


def main():

    #connect Robobo camera and choose velocity
    print("Starting test app")
    rob.connect()
    video = RoboboVideo(IP)
    video.connect()
    rob.startStream()
    print("Showing images")
    rob.moveWheels(20, 20)

    while True:
        inicio = time.time()
        cv2_image = video.getImage()
        flip=cv2.flip(cv2_image,1)
        imag = np.uint8(flip)
        orig = imag.copy()
        imag_blue = imag.copy()
        imag_red = imag.copy()

        cnts_blue_list, label_blue_list = identify_blue(imag_blue, clf_blue)
        cnts_red_list, label_red_list = identify_red(imag_red, clf_red)
        fin = time.time()
        #print("tiemo en procesar", fin - inicio)
        print("Labels (red):", label_red_list)
        print("Labels (blue):", label_blue_list)
        #time.sleep(0.1) adjust frames per second
    rob.stopMotors()
    rob.stopStream()
    video.disconnect()

"""
MAIN PROGRAM
"""
if __name__ == "__main__":

    #load models
    clf_red = load("mlp_red.joblib")
    clf_blue = load("mlp_blue.joblib")

    IP = "10.113.36.249"
    rob = Robobo(IP)
    main()
