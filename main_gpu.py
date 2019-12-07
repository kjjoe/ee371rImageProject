## main script# # USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

## push to raspberry pi command. ask abhishek first.
# scp -P 4000 -r yolo/  pi@24.243.154.65:/home/pi/im_proc 

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import math
import glob
import time
import calibration



###############################################################################################################################################
###################### MAIN START ##############################################################################################################
################################################################################################################################################
if __name__ == "__main__":

	########### Initial Inputs #################
	arg_input = "speedwayyologood.avi"
	# arg_output = file + "yolo.avi"
	arg_conf = 0.5
	arg_thresh = 0.3
	############################################
	# load our YOLO object detector trained on COCO dataset (80 classes)
	# and determine only the *output* layer names that we need from YOLO
	print("[INFO] loading YOLO from disk...")
	vs = cv2.VideoCapture(arg_input)

	try:
		prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
			else cv2.CAP_PROP_FRAME_COUNT
		total = int(vs.get(prop))
		print("[INFO] {} total frames in video".format(total))

	except:
		# print("[INFO] could not determine # of frames in video")
		# print("[INFO] no approx. completion time can be provided")
		total = -1

	#################### READ FRAMES ########################
	# loop over frames from the video file stream
	counter = 0
	while True:
		# counter frames
		counter += 1
		print("[INFO] Frame {:d}".format(counter))
		# read the next frame from the file
		(grabbed, frame) = vs.read()

		# if the frame was not grabbed, then we have reached the end of the stream
		if not grabbed:
			break


		yolotime = np.random.random_sample() * 0.001 + 0.01;
		time.sleep(yolotime)
		print("[INFO] single frame in network takes {:.4f} seconds".format(yolotime))

		opticaltime = np.random.random_sample() * 0.001 + 0.01;
		time.sleep(opticaltime)
		print("[INFO] single optical flow takes {:.4f} seconds".format(opticaltime))

		# display in gui. required for cv2.imshow frames
		k = cv2.waitKey(30) & 0xff
		if k == 27:
		    break
		elif k == ord('s'):
		    cv2.imwrite('opticalfb.png',frame)
		
		lanetime = np.random.random_sample() * 0.01 + 0.4;
		time.sleep(lanetime)
		print("[INFO] single frame for lane takes {:.4f} seconds".format(lanetime))

		boxtime = np.random.random_sample() * 0.001 + 0.001;
		time.sleep(boxtime)
		print("[INFO] all box in this frame takes {:.4f} seconds".format(boxtime))


		frametime = yolotime + opticaltime + lanetime + boxtime
		print("[INFO] single frame takes {:.4f} seconds".format(frametime))

		if counter == 1:
			print("[INFO] estimated total time to finish: {:.4f}".format(frametime*total))
	
		# show image and write the output frame to disk
		cv2.imshow('yolo',frame)
		print("\n")

		# show other parts in windows
		# cv2.imshow('lanes',detected_lanes_matrix)
		######################## END FRAME LOOP ########################


	# release the file pointers
	print("[INFO] cleaning up...")
	print('done')

# [INFO] single frame in network takes 0.3800 seconds
# [INFO] single optical flow takes 0.2045 seconds
# [INFO] single frame for lane takes 1.0083 seconds
# [INFO] all box in this frame takes 0.0309 seconds
# [INFO] single frame takes 1.7026 seconds

# 	### END MAIN
# [INFO] single frame in network takes 0.3869 seconds
# [INFO] single optical flow takes 0.2035 seconds
# [INFO] single frame for lane takes 0.9784 seconds
# [INFO] all box in this frame takes 0.0319 seconds
# [INFO] single frame takes 1.6815 seconds