import cv2
import glob
import time
import numpy as np

def ColorTrack(frame):
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_color = np.array([110, 50, 50])
	upper_color = np.array([130, 255, 255])

	mask = cv2.inRange(hsv, lower_color, upper_color)

	res = cv2.bitwise_and(frame, frame, mask=mask)

	return (res, mask)

def draw(img, corners, imgpts):
	corner = tuple(corners[0].ravel())
	img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
	img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
	img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
	return img

if __name__ == "__main__":
	cap = cv2.VideoCapture(0)
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp = np.zeros((6*7,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
	axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

	objpointes = []
	imgpoints = []

	#print objp

	try:
		calib_count = 0
		mtx, dist = None, None
		while True:

			ret, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
			if True:
				ret, corners = cv2.findChessboardCorners(gray, (7,6), None)
				if ret == True:
	
					corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
					if calib_count < 20:
						objpointes.append(objp)
						imgpoints.append(corners)

						cv2.drawChessboardCorners(frame, (7,6), corners, ret)
					elif calib_count == 20:
						ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpointes, imgpoints, gray.shape[::-1], None, None)
					else:
						ret,rvecs, tvecs, inliers = cv2.solvePnP(objp, corners, mtx, dist)
						imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
						frame = draw(frame, corners2, imgpts)
					calib_count += 1
					print calib_count
			

			cv2.imshow('Video', frame)
			cv2.imshow('Gray', gray)
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
	except Exception, e:
		print str(Exception), e
	cap.release()
	cv2.destroyAllWindows()