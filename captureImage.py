import cv2
import glob
import time
import os
import numpy as np


if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	try:
		calib_count = 0
		mtx, dist = None, None
		while True:

			ret, frame = cap.read()
			
			cv2.imshow('Video', frame)
			
			key = cv2.waitKey(30)
			if  key & 0xFF == ord('c'):
				print "capturing"
				cv2.imwrite(os.path.join(os.path.join(os.getcwd(),'data'),str(time.time()) + '_chessboard.jpg'), frame)

			if key & 0xFF == ord('q'):
				break
	except Exception, e:
		print str(Exception), e
	cap.release()
	cv2.destroyAllWindows()