import cv2
import mediapipe as mp
import time

class poseDetector():
    def __init__(self, mode = False, upBody = False, smooth = True, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.upBody = upBody
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=1,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                # for id, lm in enumerate(results.pose_landmarks.landmark):
                #     h, w, c = img.shape
                #     cx, cy = int(lm.x*w), int(lm.y*h)
                #     print(id, lm)
                #     cx, cy = int(lm.x*w), int(lm.y*h)
                #     cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return img

    def lmPosition(self, img, draw=True):
        lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        return lmList

def main():
    cap = cv2.VideoCapture(0)  # Open the webcam
    detector = poseDetector()

    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.lmPosition(img)
        if lmList:
            cv2.circle(img, (lmList[7][1], lmList[7][2]), 15, (0, 255, 0), cv2.FILLED)
        cv2.imshow("Image", img)

        if cv2.waitKey(1) & 0xFF == ord('X'):
            break



if __name__ == "__main__":
    main()


