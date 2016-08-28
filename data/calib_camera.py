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

def draw2(img, corners, imgpts):
	print type(img)
	corner = tuple(corners[0].ravel())
	cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
	cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
	cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
	print img.shape
	return img

def draw(img, corners, imgpts):
	imgpts = np.int32(imgpts).reshape(-1,2)
	cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)

	for i,j in zip(range(4),range(4,8)):
		cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)

	cv2.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)

	return img

if __name__ == "__main__":

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	objp = np.zeros((6*7,3), np.float32)
	objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
	#axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

	axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0], [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

	objpoints = []
	imgpoints = []

	images = glob.glob("*.jpg")

	for fname in images:
		img = cv2.imread(fname)
		#print type(img)
		#print img.shape
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

		if ret == True:
			objpoints.append(objp)

			cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners)

			cv2.drawChessboardCorners(img, (7,6), corners,ret)
			#if img != None:
			cv2.imshow('img', img)
			cv2.imshow('Gray', gray)
			cv2.waitKey(70)

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

	#print ret,mtx,dist,rvecs, tvecs

	#img = cv2.imread('amit.image')
	#h, w = img.shape[:2]

	#newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

	#dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

	#cv2.imwrite('calibresult.png', dst)

	cap = cv2.VideoCapture(0)

	try:
		while True:
			ret, img = cap.read()
			#img = cv2.imread(fname)
			#print type(img)
			#print img.shape
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	
			ret, corners = cv2.findChessboardCorners(gray, (7,6),None)
	
			if ret == True:
	
				cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
	
				rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
	
				imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	
				img = draw(img,corners,imgpts)
	
			cv2.imshow('img', img)
			cv2.imshow('Gray', gray)
			key = cv2.waitKey(30)
			if key &  0xFF == ord('q'):
				break
	except Exception as e:
		print e
	cap.release()
	cv2.destroyAllWindows()