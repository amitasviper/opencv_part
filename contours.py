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
		
		kernel_size = 5
		kernel = np.ones((kernel_size, kernel_size), np.float32)/kernel_size**2

		#blurFrame = cv2.filter2D(frame, -1, kernel)

		blurFrame = cv2.bilateralFilter(frame, 9, 75, 75)

		hsv = cv2.cvtColor(blurFrame, cv2.COLOR_BGR2HSV)

		lower_color = np.array([95, 70, 70])
		upper_color = np.array([110, 255, 255])

		mask = cv2.inRange(hsv, lower_color, upper_color)

		opening = cv2.erode(mask, kernel, iterations=1)

		closing = cv2.dilate(opening, kernel, iterations=1)

		contours, hie = cv2.findContours(closing, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		areas = [cv2.contourArea(c) for c in contours]
		if len(areas) > 0:
			max_index = np.argmax(areas)
			cnt = contours[max_index]
		
			cv2.drawContours(frame, [cnt], -1, (0,0,255), 3)

			x,y,w,h = cv2.boundingRect(cnt)

			cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
		
		#one
		#res = cv2.bitwise_and(frame, frame, mask=mask)

		cv2.imshow('Video', frame)
		
		##one
		#cv2.imshow('Track', res)
		
		cv2.imshow('Mask', mask)
		
		#cv2.imshow("Blur", blurFrame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()