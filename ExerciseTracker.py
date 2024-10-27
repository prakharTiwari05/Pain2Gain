import time
import mediapipe as mp
import cv2
import PoseTrackingModule as ptm
import math
import streamlit as st

# Initialize Streamlit app
st.title("Welcome to Pain2Gain")
frame_placeHolder = st.empty()

# Add "X" and "✓" buttons for controlling the camera
terminate_button = st.button("❌ Stop Camera")
start_button = st.button("✅ Start Camera")

# Initialize variables
detector = ptm.poseDetector()  # Assuming poseDetector is defined in PoseTrackingModule
stagePushups = 'None'
stageSitups = 'None'
stageSquats = 'None'
counterPushups = 0
counterSitups = 0
counterSquats = 0
maxDistPushups = 0
maxDistSitups = 0
maxDistSquats = 0

# Main logic to handle the camera stream based on button inputs
cap = None  # Initialize the capture object outside the loop

# Run only if the start button is pressed
if start_button:
    cap = cv2.VideoCapture(0)  # Open the webcam

while cap and cap.isOpened():
    # Check if the stop button is pressed
    if terminate_button:
        cap.release()  # Stop the camera feed
        break

    success, img = cap.read()
    if not success:
        st.warning("No camera detected. Please connect your camera.")
        break

    img = detector.findPose(img)
    lmList = detector.lmPosition(img) if detector.results.pose_landmarks else []

    if lmList:
        # Extract landmark positions and calculate required distances and angles
        x1Eye, y1Eye = lmList[1][1], lmList[1][2]
        x2Hand, y2Hand = lmList[20][1], lmList[20][2]
        handAbove = y2Hand < y1Eye

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

        if normalDistancePushups < 0.4 and not handAbove:
            stagePushups = "down"
        if normalDistancePushups > 0.6 and stagePushups == "down" and abs(slope) < 0.6 and not handAbove:
            stagePushups = "up"
            counterPushups += 1

        # Situps detection logic
        x1Situps, y1Situps = lmList[0][1], lmList[0][2]
        x2Situps, y2Situps = lmList[25][1], lmList[25][2]
        distanceSitups = math.sqrt((x2Situps - x1Situps) ** 2 + (y2Situps - y1Situps) ** 2)

        if distanceSitups > maxDistSitups:
            maxDistSitups = distanceSitups
        normalDistanceSitups = distanceSitups / maxDistSitups if maxDistSitups != 0 else 0

        if normalDistanceSitups > 0.8 and not handAbove:
            stageSitups = "down"
        if normalDistanceSitups < 0.6 and stageSitups == "down" and abs(slope) < 4 and not handAbove:
            stageSitups = "up"
            counterSitups += 1

        # Squats detection logic
        x1Squats, y1Squats = lmList[13][1], lmList[13][2]
        x2Squats, y2Squats = lmList[25][1], lmList[25][2]
        yDistSquats = abs(y1Squats - y2Squats)

        if yDistSquats > maxDistSquats:
            maxDistSquats = yDistSquats
        normalDistanceSquats = yDistSquats / maxDistSquats if maxDistSquats != 0 else 0

        if normalDistanceSquats < 0.4 and not handAbove and abs(slope) > 6:
            stageSquats = "down"
        if normalDistanceSquats > 0.65 and stageSquats == "down" and abs(slope) > 1 and not handAbove:
            stageSquats = "up"
            counterSquats += 1

        # Overlay information on image
        cv2.line(img, (x1Shoulder, y1Shoulder), (x2Toe, y2Toe), (0, 255, 0), 4)
        cv2.putText(img, "Pushups: " + str(counterPushups), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "Situps: " + str(counterSitups), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(img, "Squats: " + str(counterSquats), (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Convert the image color from BGR to RGB for Streamlit display
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_placeHolder.image(img_rgb, channels="RGB")

# Release resources if the stop button was pressed
if cap:
    cap.release()
    cv2.destroyAllWindows()
