import time
import mediapipe as mp
import cv2
import PoseTrackingModule as ptm
import math
#import pyttsx3

cap = cv2.VideoCapture(0)  # Open the webcam
detector = ptm.poseDetector()  # Assuming poseDetector is defined in HandTrackingModule
stagePushups = 'None'
stageSitups = 'None'
stageSquats = 'None'

counterPushups = 0
counterSitups = 0
counterLunges = 0
counterSquats = 0

maxDistPushups = 0
maxDistSitups = 0
maxDistSquats = 0
maxDistThrottle = 0

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.lmPosition(img) if detector.results.pose_landmarks else []  # Safeguard for pose landmarks
    if lmList:
        # Body slope detection
        x1Eye, y1Eye = lmList[1][1], lmList[1][2]
        x2Hand, y2Hand = lmList[20][1], lmList[20][2]
        handAbove = True;
        if(abs(y2Hand) > abs(y1Eye)):
            handAbove = False;
        else:
            handAbove = True;

        x1Shoulder, y1Shoulder = lmList[11][1], lmList[11][2]
        x2Toe, y2Toe = lmList[27][1], lmList[27][2]
        slope = (y1Shoulder - y2Toe) / (x1Shoulder - x2Toe) if x2Toe != x1Shoulder else 0

        # Pushups detection logic
        x1Pushups, y1Pushups = lmList[11][1], lmList[11][2]
        x2Pushups, y2Pushups = lmList[19][1], lmList[19][2]
        distancePushups = y2Pushups - y1Pushups

        if distancePushups > maxDistPushups:
            maxDistPushups = distancePushups

        normalDistancePushups = distancePushups / maxDistPushups if maxDistPushups != 0 else 0

        if normalDistancePushups < 0.4 and handAbove == False:
            stagePushups = "down"
        if normalDistancePushups > 0.6 and stagePushups == "down" and abs(slope) < 0.6 and handAbove == False:
            stagePushups = "up"
            counterPushups += 1
            #text_speech.say("A pushup")

        # Situps detection logic
        x1Situps, y1Situps = lmList[0][1], lmList[0][2]
        x2Situps, y2Situps = lmList[25][1], lmList[25][2]
        distanceSitups = math.sqrt((x2Situps - x1Situps) ** 2 + (y2Situps - y1Situps) ** 2)

        if distanceSitups > maxDistSitups:
            maxDistSitups = distanceSitups

        normalDistanceSitups = distanceSitups / maxDistSitups if maxDistSitups != 0 else 0

        if normalDistanceSitups > 0.8 and handAbove == False:
            stageSitups = "down"
        if normalDistanceSitups < 0.6 and stageSitups == "down" and abs(slope) < 4 and handAbove==False:
            stageSitups = "up"
            counterSitups += 1

        # Squats detection logic
        x1Squats, y1Squats = lmList[13][1], lmList[13][2]
        x2Squats, y2Squats = lmList[25][1], lmList[25][2]
        yDistSquats = abs(y1Squats - y2Squats)

        if yDistSquats > maxDistSquats:
            maxDistSquats = yDistSquats

        normalDistanceSquats = yDistSquats / maxDistSquats if maxDistSquats != 0 else 0
        if normalDistanceSquats < 0.4 and handAbove == False and abs(slope) > 6 :
            stageSquats = "down"
        if normalDistanceSquats > 0.65 and stageSquats == "down" and abs(slope) > 1 and handAbove == False:
            stageSquats = "up"
            counterSquats += 1

        # Display and visualization
        cv2.line(img, (x1Shoulder, y1Shoulder), (x2Toe, y2Toe), (0, 255, 0), 4)

    # Display counter and throttle distance
    cv2.putText(img, "Pushups: " + str(counterPushups), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Slope: " + str(abs(slope)), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Situps: " + str(counterSitups), (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, "Squats: " + str(counterSquats), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
