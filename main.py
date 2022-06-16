# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from PIL import Image

import pytesseract
from pytesseract import Output
import argparse
import cv2
from PIL import ImageGrab
import pyautogui
import numpy as np

def getImage():
    width, height = pyautogui.size()
    print(width)
    #img = ImageGrab.grab(bbox=(0, 0, width*2, height*2))
    img = pyautogui.screenshot()
    img_np = np.array(img)
    frame = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)
    return frame


def centroid(x, y, w, h):
    return x + int(w/2), y + int(h/2)

def runProg():
    # construct the argument parser and parse the arguments

    ap = argparse.ArgumentParser()
    #ap.add_argument("-i", "--image", required=True,
     #               help="path to input image to be OCR'd")
    ap.add_argument("-c", "--min-conf", type=int, default=0,
                    help="mininum confidence value to filter weak text detection")
    args = vars(ap.parse_args())

    #image = cv2.imread('file/test.png')
    #rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = getImage()
    results = pytesseract.image_to_data(image, output_type=Output.DICT)
    for i in range(0, len(results["text"])):
        # extract the bounding box coordinates of the text region from
        # the current result
        x = results["left"][i]
        y = results["top"][i]
        w = results["width"][i]
        h = results["height"][i]
        # extract the OCR text itself along with the confidence of the
        # text localization
        text = results["text"][i]
        conf = int(float(results["conf"][i]))

        # filter out weak confidence text localizations
        if conf > args["min_conf"]:
            # display the confidence and text to our terminal
            print("Confidence: {}".format(conf))
            print("Text: {}".format(text))
            print("")
            # strip out non-ASCII text so we can draw the text on the image
            # using OpenCV, then draw a bounding box around the text along
            # with the text itself
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(image, centroid(x, y, w, h), radius=3, color=(0, 0, 255), thickness=2)
            cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        .5, (0, 0, 255), 1)
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    runProg()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
