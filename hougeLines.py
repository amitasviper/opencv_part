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
	font = cv2.FONT_HERSHEY_SIMPLEX
	try:
		while True:
			#print "Yes"
			ret, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			edges = cv2.Canny(gray, 10, 250, apertureSize=3)
			#lines = cv2.HoughLines(edges, 10, np.pi/1800, 50)
			minLineLength = 100
			maxLineGap = 10
			lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
			if lines != None:
				for line in lines:
					x1,y1,x2,y2 = line[0]
					cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
			cv2.imshow('Video', frame)
			cv2.imshow('Gray', gray)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	except Exception as e:
		print e
	cap.release()
	cv2.destroyAllWindows()