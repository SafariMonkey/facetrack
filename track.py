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


def XYWH_to_LTRB(rect):
    ''' Convert a left-top-width-height rectangle to a left-top-right-bottom rectangle. '''
    rX, rY, rW, rH = rect
    return rX, rY, rX + rW, rY + rH

def LTRB_to_XYWH(rect):
    ''' Convert a left-top-right-bottom rectangle to a left-top-width-height rectangle. '''
    rL, rT, rR, rB = rect
    return rL, rT, rR - rL, rB - rT

def expand_rect(rect, amount, bounds):
    rL, rT, rR, rB = XYWH_to_LTRB(rect)
    bL, bT, bR, bB = XYWH_to_LTRB(bounds)
    return LTRB_to_XYWH((
        max(rL-amount, bL),
        max(rT-amount, bT),
        min(rR+amount, bR),
        min(rB+amount, bB)
    ))


def main():
    cap_w, cap_h = 2560/2, 960/2

    start_time = time.time()
    cap = cv2.VideoCapture(2)
    assert set_res(cap, cap_w, cap_h) == (str(cap_w), str(cap_h))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 0)


    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_5_face_landmarks.dat')

    lowest_latency = get_latency(cap, start_time, time.time())
    print("Starting latency: {}".format(lowest_latency))

    last_rects = {'left': [], 'right': []}
    rois = {'left': [], 'right': []}

    frame_number = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            continue
        frame_number += 1
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
        full_frame = (0, 0, width, height)

        for side, image in [('left', left), ('right', right)]:
            orig_image = image.copy()

            if not rois[side]:
                rois[side] = [full_frame]

            for roi in rois[side]:
                roiL, roiT, roiR, roiB = XYWH_to_LTRB(roi)
                cv2.rectangle(image, (roiL, roiT), (roiR, roiB),
                              (30, 50, 70), 1)

            roi_images = [((x, y), image[y:y+h, x:x+w]) for (x, y, w, h) in rois[side]]
            cv2.imshow(f'roi_{side}', roi_images[0][1])

            def detect_upscale_fallback(frame, max_upscale):
                for scale in range(max_upscale):
                    rects = detector(frame, scale)
                    if rects:
                        break
                return rects

            new_rects = []
            for offset, roi_image in roi_images:
                width, height, _ = roi_image.shape
                if max(width, height) < 100:
                    max_upscale = 4
                elif max(width, height) < 200:
                    max_upscale = 3
                else:
                    max_upscale = 2
                rects = detect_upscale_fallback(roi_image, max_upscale)
                if rects:
                    rectL, rectT, rectR, rectB = XYWH_to_LTRB(face_utils.rect_to_bb(rects[0]))
                    cv2.rectangle(roi_image, (rectL, rectT), (rectR, rectB),
                            (0, 255, 0), 1)
                    rects = (face_utils.rect_to_bb(rect) for rect in rects)
                    o_x, o_y = offset
                    rects = ((x+o_x, y+o_y, w, h) for (x, y, w, h) in rects)
                    new_rects += rects
            if new_rects:
                rois[side] = new_rects
            rois[side] = [expand_rect(roi, 15, bounds=full_frame)
                          for roi in rois[side]]
            rects = new_rects

            # stash the rectected
            if len(rects):
                last_rects[side] = rects
                fresh = True
            else:
                rects = last_rects[side]
                fresh = False

            for rect in rects:
                # compute the bounding box of the face and draw it on the
                # frame
                rectL, rectT, rectR, rectB = XYWH_to_LTRB(rect)
                cv2.rectangle(image, (rectL, rectT), (rectR, rectB),
                            (0, 255, 0) if fresh else (0, 50, 0), 1)

                # determine the facial landmarks for the face region, then
                # convert the facial landmark (x, y)-coordinates to a NumPy
                # array
                shape = predictor(orig_image, dlib.rectangle(*XYWH_to_LTRB(rect)))
                shape = face_utils.shape_to_np(shape)
        
                # loop over the (x, y)-coordinates for the facial landmarks
                # and draw each of them
                for (i, (x, y)) in enumerate(shape):
                    cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
                    cv2.putText(image, str(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # Display the resulting frame
        cv2.imshow('frame', np.concatenate((left, right), axis=1))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
