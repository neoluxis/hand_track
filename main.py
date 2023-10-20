import cv2 as cv
import mediapipe as mp
import time

pTime = 0
cTime = 0

cam = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

if __name__ == "__main__":
    while True:
        success, img = cam.read()

        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if (results.multi_hand_landmarks):
            for hand in results.multi_hand_landmarks:
                for id, lm in enumerate(hand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    print(id, cx, cy)
                    cv.putText(img, str(id), (cx, cy), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

                mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv.putText(img, str(int(fps)), (10, 70), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv.imshow("Image", img)
        if cv.waitKey(1) == ord('q'):
            break
