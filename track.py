import numpy as np
import cv2
import dlib
import itertools
import imutils
from imutils import face_utils

import time

def get_latency(cap, start_time, now):
    return int(round((now - start_time) * 1000)) - cap.get(cv2.CAP_PROP_POS_MSEC)

def set_res(cap, x, y):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(x))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(y))
    return str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def main():
    cap_w, cap_h = 2560/2, 960/2

    start_time = time.time()
    cap = cv2.VideoCapture(2)
    assert set_res(cap, cap_w, cap_h) == (str(cap_w), str(cap_h))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')

    last_left_rects, last_right_rects = [], []

    lowest_latency = get_latency(cap, start_time, now)
    print("Starting latency: {}".format(lowest_latency))

    mode = 'search'

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            continue
        now = time.time()
        latency = get_latency(cap, start_time, now)
        if latency < lowest_latency:
            lowest_latency = latency
            print("New lowest latency: {}".format(lowest_latency))
        elif latency > (lowest_latency+34):
            # skip forward
            continue

        height, width, channels = frame.shape
        crop = (width//2 - height)//2
        left, right = frame[:, crop:(width//2-crop)], frame[:, (width//2+crop):-crop]
        
        height, width, channels = left.shape
        frame = np.concatenate((left, right), axis=1)

        def detect_upscale_fallback(frame):
            for scale in range(2):
                rects = detector(frame, scale)
                if rects :
                    break
            return rects

        left_rects = detect_upscale_fallback(left)
        right_rects = detect_upscale_fallback(right)

        # stash the rectected
        if len(left_rects):
            last_left_rects = left_rects
            left_fresh = True
        else:
            left_rects = last_left_rects
            left_fresh = False
        if len(right_rects):
            last_right_rects = right_rects
            right_fresh = True
        else:
            right_rects = last_right_rects
            right_fresh = False

        for image, rect, fresh, offset in itertools.chain(
                ((left, rect, left_fresh, 0) for rect in left_rects),
                ((right, rect, right_fresh, width) for rect in right_rects)):
            # compute the bounding box of the face and draw it on the
            # frame
            (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
            bX += offset
            cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),
                        (0, 255, 0) if fresh else (0, 50, 0), 1)


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

if __name__ == "__main__":
    main()
