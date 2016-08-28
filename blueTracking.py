import cv2
import numpy as np

def ColorTrack(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_color = np.array([110, 50, 50])
	upper_color = np.array([130, 255, 255])

	mask = cv2.inRange(hsv, lower_color, upper_color)

	res = cv2.bitwise_and(frame, frame, mask=mask)

	return (res, mask)

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	ret, frame = cap.read()
	#print dir(frame)
	font = cv2.FONT_HERSHEY_SIMPLEX
	while True:
		ret, frame = cap.read()
		#cv2.line(frame, (0,0), (50,50), (255,0,0), 4)
		#cv2.putText(frame, "Amit Yadav", (20,300), font, 2, (255,65,0),5, 2)
		
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

		lower_color = np.array([95, 70, 70])
		upper_color = np.array([110, 255, 255])

		mask = cv2.inRange(hsv, lower_color, upper_color)

		res = cv2.bitwise_and(frame, frame, mask=mask)

		cv2.imshow('Video', frame)
		cv2.imshow('Track', res)
		cv2.imshow('Mask', mask)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()