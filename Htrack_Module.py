import cv2
import mediapipe as mp
import time


class HandDetector:
	def __init__(self, mode=False, max_hands=2, min_det_conf=0.5, min_track_conf=0.5):
		self.img_mode = mode
		self.img_hands = max_hands
		self.det_conf = min_det_conf
		self.track_conf = min_track_conf

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.img_mode, self.img_hands, self.det_conf, self.track_conf)
		self.mpDraw = mp.solutions.drawing_utils

	def find_hands(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		self.res = self.hands.process(imgRGB)

		if self.res.multi_hand_landmarks:
			for handldmarks in self.res.multi_hand_landmarks:
				if draw:
					self.mpDraw.draw_landmarks(img, handldmarks, self.mpHands.HAND_CONNECTIONS)
		return img

	def find_position(self, img, handNo=0, draw=True):
		landmarklist = []

		if self.res.multi_hand_landmarks:
			myHand = self.res.multi_hand_landmarks[handNo]
			for id, lm in enumerate(myHand.landmark):
				h, w, c = img.shape
				cx, cy = int(lm.x*w), int(lm.y*h)
				landmarklist.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 7, (3, 248, 255), cv2.FILLED)
		return landmarklist


def main():
	ptime, ctime = 0, 0
	cap = cv2.VideoCapture(0)
	detector = HandDetector()
	while True:
		ret, frame = cap.read()
		frame = detector.find_hands(frame)

		flip = cv2.flip(frame, 1)

		ctime = time.time()
		fps = 1 / (ctime - ptime)
		ptime = ctime

		cv2.putText(flip, f"FPS:{int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
		cv2.imshow("Video", flip)
		cv2.waitKey(1)


if __name__ == "__main__":
	main()