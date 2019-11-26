# # USAGE
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

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,
# 	help="path to input video")
# ap.add_argument("-o", "--output", required=True,
# 	help="path to output video")
# ap.add_argument("-y", "--yolo", required=True,
# 	help="base path to YOLO directory")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-t", "--threshold", type=float, default=0.3,
# 	help="threshold when applyong non-maxima suppression")
# args = vars(ap.parse_args())
file = "150"
arg_input = "videos/" + file + ".mp4"
# arg_input = "videos/project_video.mp4"
arg_output = file + "yolo.avi"
arg_yolo = "yolo-coco"
arg_conf = 0.5
arg_thresh = 0.3


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([arg_yolo, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([arg_yolo, "yolov3.weights"])
configPath = os.path.sep.join([arg_yolo, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(arg_input)
writer = None
(W, H) = (None, None)

# initialization for optical flow
ret, frame1 = vs.read() # read initial frame
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)   
hsv[...,1] = 255
writer2 = None

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	# print("[INFO] could not determine # of frames in video")
	# print("[INFO] no approx. completion time can be provided")
	total = -1

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > arg_conf:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, arg_conf,
		arg_thresh)

	# pause on yolo

	############ optical flow
	next = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	# calculate optical flow
	flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
	mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])  # convert to polar coord

	hsv[...,0] = ang*180/np.pi/2                          		# set angle to hue 
	hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)  # set magnitue to value
	rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

	size = mag.shape

	# display in gui
	
	k = cv2.waitKey(30) & 0xff
	if k == 27:
	    break
	elif k == ord('s'):
	    cv2.imwrite('opticalfb.png',frame)
	    cv2.imwrite('opticalhsv.png',rgb)
	prvs = next

	# write to video
	if writer2 is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer2 = cv2.VideoWriter(file + "optical.avi", fourcc, 30,(frame.shape[1], frame.shape[0]), True)
	# write the output frame to disk
	writer2.write(rgb)

	#### resume yolo and bounding box. include information on optical flow
	# ensure at least one detection exists

	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])

			# draw a bounding box rectangle and label on the frame
			color = [int(c) for c in COLORS[classIDs[i]]]

			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "{}: {:.4f}".format(LABELS[classIDs[i]],
				confidences[i])
			cv2.putText(frame, text, (x, y - 5),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

			## include information on optical flow

			mag_box = mag[y:y+h,x:x+w] # bound this from 0 to max size
			avg = np.mean(mag_box)
			if math.isnan(avg):
				x_max = x+w;
				y_max = y+h;

				if x < 0: 
					x = 0
				if y < 0:
					y = 0
				if x+w > size[0]:
					x_max = size[0]
				if y+h > size[1]:
					y_max = size[1]
				mag_box = mag[y:y_max,x:x_max]
				avg = np.mean(mag_box)

			text_opt = "Flow: {:0.4f}".format(avg)
			cv2.putText(frame, text_opt, (x, y + 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], 2)

			## put information on optical flow as well
			cv2.rectangle(rgb, (x, y), (x + w, y + h), color, 2)
			cv2.putText(rgb, text_opt, (x, y + 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,255,0], 2)




	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(arg_output, fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	cv2.imshow('yolo',frame)
	cv2.imshow('optical',rgb)
	# write the output frame to disk
	writer.write(frame)


# release the file pointers
print("[INFO] cleaning up...")
writer.release()
writer2.release()
vs.release()