import cv2
import time
import Htrack_Module
import math
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

pre_t, c_time = 0, 0
wcam, hcam = 800, 600

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol = volRange[0], volRange[1]

vol, volBar, volBarPer = 0, 400, 0

detection = Htrack_Module.HandDetector(min_det_conf=0.8)

while True:
	val, frame = cap.read()
	fm_flip = cv2.flip(frame, 1)

	fm_flip = detection.find_hands(fm_flip)
	lmlist = detection.find_position(fm_flip, draw=False)
	if len(lmlist) != 0:
		x1, y1 = lmlist[4][1], lmlist[4][2]
		x2, y2 = lmlist[8][1], lmlist[8][2]
		cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

		cv2.circle(fm_flip, (x1, y1), 12, (3, 248, 255), cv2.FILLED)
		cv2.circle(fm_flip, (x2, y2), 12, (3, 248, 255), cv2.FILLED)
		cv2.line(fm_flip, (x1, y1), (x2, y2), (3, 248, 255), 3)
		cv2.circle(fm_flip, (cx, cy), 8, (3, 248, 255), cv2.FILLED)

		length = math.hypot((x2 - x1), (y2 - y1))

		vol = np.interp(length, [30, 150], [minVol, maxVol])
		volBar = np.interp(length, [30, 150], [400, 150])
		volBarPer = np.interp(length, [30, 150], [0, 100])
		print(vol)
		volume.SetMasterVolumeLevel(vol, None)

		if length < 30:
			cv2.circle(fm_flip, (cx, cy), 8, (255, 0, 255), cv2.FILLED)

	cv2.rectangle(fm_flip, (50, 150), (85, 400), (198, 212, 78), 3)
	cv2.rectangle(fm_flip, (50, int(volBar)), (85, 400), (198, 212, 78), cv2.FILLED)
	cv2.putText(fm_flip, f"{int(volBarPer)}%", (50, 430), cv2.FONT_HERSHEY_COMPLEX, 1, (198, 212, 78), 3)

	c_time = time.time()
	fps = 1 / (c_time - pre_t)
	pre_t = c_time

	cv2.putText(fm_flip, f"FPS:{int(fps)}", (30, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

	cv2.imshow("Cam-01", fm_flip)
	cv2.waitKey(1)
