import cv2
import numpy as np
import time
import pyautogui
import argparse
import mediapipe as mp
import copy

from utils.helper import *
from utils.cvfpscalc import CvFpsCalc

from nn.classifier_hand_lm import KeyPointClassifier, PointHistoryClassifier
from collections import Counter
from collections import deque

# Globals
wCam, hCam = 640, 480
frameR = 100     #Frame Reduction
smoothening = 7  #random value
keypoint_classifier_labels = ['Open', 'Close', 'Pointer', 'OK']
point_history_classifier_labels = ['Stop', 'Clockwise', 'Counterclockwise', 'Move']

def main(args):
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity = 0,
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history
    finger_gesture_history = deque(maxlen=history_length)

    mode = 0
    pTime = 0
    plocX, plocY = 0, 0
    clocX, clocY = 0, 0

    wScr, hScr = pyautogui.size()
    print(f"Screen width {wScr} Screen height {hScr}")

    clicked = False
    dragging = False
    use_brect = True

    while True:
        fps = cvFpsCalc.get()
        # Capture image
        ret, image = cap.read()
        if not ret:
            break
        image = cv2.flip(image, 1)
        debug_image = copy.deepcopy(image)

        # Detect hands and landmarks with mediapipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                x1 = landmark_list[8][0]
                y1 = landmark_list[8][1]   

                x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))

                # smooth values
                clocX = plocX + (x3 - plocX) / smoothening
                clocY = plocY + (y3 - plocY) / smoothening

                if keypoint_classifier_labels[hand_sign_id] == 'Pointer':

                    # move mouse pointer
                    pyautogui.moveTo(wScr - clocX, clocY, duration=0, _pause=False)
                    cv2.circle(debug_image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                else:

                    if keypoint_classifier_labels[hand_sign_id] == 'Close':
                        # press mouse button for dragging
                        if not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                            print("start dragging")                            
                            cv2.circle(debug_image, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
                        else:
                            pyautogui.moveTo(wScr - clocX, clocY, duration=0, _pause=False)
                            cv2.circle(debug_image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)                            
                    else:
                        # release mouse button
                        if dragging:                        
                            pyautogui.mouseUp()
                            dragging = False
                            print("stop dragging")        

                # clockwise pointer movement - single click
                if point_history_classifier_labels[most_common_fg_id[0][0]] == 'Clockwise':
                    if not clicked:
                        cv2.circle(debug_image, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                        pyautogui.click()
                        clicked = True
                        print("clicking", x1, y1)
                else:
                    clicked = False

                plocX, plocY = clocX, clocY
                # Drawing part
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )
        else:
            point_history.append([0, 0])

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, 0)

        # Display window with video feed
        cv.imshow('Hand Gesture Recognition', debug_image)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)   
    args = parser.parse_args()

    main(args)