import cv2
import mediapipe as mp
import numpy as np

class HandDetector:
    def __init__(self, mode=False, maxhands=2,modecomplex=1, detectioncon=0.5,trackcon=0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.modecomplex = modecomplex
        self.detectioncon = detectioncon
        self.trackcon=trackcon

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(self.mode, self.maxhands, self.modecomplex,self.detectioncon,self.trackcon)

    def detect(self, image_path):
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        self.results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            x,y,w,h = cv2.boundingRect(np.array([[l.x * width, l.y * height] for l in hand_landmarks.landmark]).astype(np.float32))
            x_cen = x + w * 0.5
            y_cen = y + h * 0.5
            bar = max(w, h) * 0.5
            up = int(y_cen - 1.1*bar)
            down = int(y_cen + 1.1*bar)
            left = int(x_cen - 1.1*bar)
            right = int(x_cen + 1.1*bar)
            if up < 0:
                up = 0
            if left < 0:
                left = 0
            if down > height:
                down = height
            if right > width:
                right = width
            hand_image = image[up:down, left:right]
            return hand_image
        else:
            return image
    def mark(self,image):
        height, width, _ = image.shape
        self.results = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.results.multi_hand_landmarks:
            hand_landmarks = self.results.multi_hand_landmarks[0]
            x, y, w, h = cv2.boundingRect(
                np.array([[l.x * width, l.y * height] for l in hand_landmarks.landmark]).astype(np.float32))
            return x,y,x+w,y+h,True
        else:
            return 0,0,width,height,False
    def close(self):
        self.hands.close()

def hand_dector(img_path):
    hand_detector = HandDetector()
    img = hand_detector.detect(img_path[0])
    hand_detector.close()
    return img
def hand_mark(img):
    hand_detector = HandDetector()
    left,up,right,down,mbool = hand_detector.mark(img)
    hand_detector.close()
    return left,up,right,down,mbool
