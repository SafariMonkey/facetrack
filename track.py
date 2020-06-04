import numpy as np
import cv2
import dlib
import itertools
import imutils
from imutils import face_utils

import time


def set_res(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


cap = cv2.VideoCapture(2)
assert set_res(cap, 2560/2, 960/2) == (str(2560.0/2), str(960.0/2))
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cap.set(cv2.CAP_PROP_EXPOSURE, 25)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')

last_left_dets, last_right_dets = None, None

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        continue

    # preserve original data
    o_height, o_width, o_depth = frame.shape
    o_left, o_right = frame[:o_height, o_width//2:], frame[:o_height, :o_width//2]

    left = imutils.resize(left, width=400)
    right = imutils.resize(right, width=400)

    height, width, depth = frame.shape
    left, right = frame[:height, width//2:], frame[:height, :width//2]

    left_rects = detector(left, 1)
    right_rects = detector(right, 1)

    # stash the rectected
    if len(left_rects):
        last_left_rects = left_rects
    if len(right_rects):
        last_right_rects = right_rects

    for image, rect, offset in itertools.chain(
            ((left, rect, width//2) for rect in left_rects),
            ((right, rect, 0) for rect in right_rects)):
        # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        bX += offset
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                      (0, 255, 0), 1)


        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(image, rect)
        shape = face_utils.shape_to_np(shape)
 
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw each of them
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x+offset, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (x+offset - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

def map_np(f, combined):
    return np.array(list(map(f, combined)))
