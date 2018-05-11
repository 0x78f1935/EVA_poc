import cv2
from skimage.measure import compare_ssim
import imutils
import datetime

# Capture webcam, change 0 to any device input
cap = cv2.VideoCapture(0)
# sensitivity of motion detection; Where 1 is sensitive and 0 Not
sensitivity = 0.94
# debug settings below
debug = False
average_check = []

while True:
    # Capture frame-by-frame
    frame_a = cap.read()[1]
    # Gray out our first frame
    grayA = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    # Capture second frame
    frame_b = cap.read()[1]
    # Gray out second frame
    grayB = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # compare Structural Similarity Index (SSIM) between the two frames
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # debug print
    if debug:
        print("SSIM: {}".format(score))

    # If image score is lower or equal to sensitivity to take action.
    # This means the difference between the two frames are major enough
    # To call it a detection of motion.
    if score <= sensitivity:
        # debug print
        if debug:
            average_check.append(score)

        # threshold the difference in the frames, followed by finding contours to
        # obtain the regions of the two input images that differ
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # compute the bounding box of the contour and then draw the
        # bounding box on both input frames to represent where the two
        # frames differ
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame_a, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame_b, (x, y), (x + w, y + h), (0, 0, 255), 1)

    # Display the frame A
    cv2.imshow("Frame: A", frame_a)

    if score <= sensitivity:
        # Create a picture and store it in the img folder.
        # The file is named the current date and time.
        cv2.imwrite('img/{}.png'.format(datetime.datetime.now().strftime("%d-%B-%Y-%I%M%S%p")), frame_a)

        # Display frame B
        cv2.imshow("Frame: B", frame_b)

        # If debug modus
        if debug:
            cv2.imshow("Diff", diff)
            cv2.imshow("Thresh", thresh)

    # if the key Q is pressed quit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # if debug print average SSIM
        if debug:
            print(min(average_check))
        # stop the camera loop
        break

    # hold space to manual take pictures with the current date and time as name.
    # Those images are also stored in the folder img
    if cv2.waitKey(32) == ord(' '):
        print('Space detected : IMG Saved')
        cv2.imwrite('img/{}.png'.format(datetime.datetime.now().strftime("%d-%B-%Y-%I%M%S%p")), frame_a)

# When everything done, release the capture, exit program
cap.release()
cv2.destroyAllWindows()
exit()