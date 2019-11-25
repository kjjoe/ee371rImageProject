import sys
import numpy as np
import cv2
import os
print(sys.version) 
print(cv2.__version__)


file = "343" # file to read

print( "yolo_video.py --input videos/" + file +  ".mp4 --output " + file + "yolo.avi --yolo yolo-coco")
# os.system("yolo_video.py --input videos/" + file +  ".mp4 --output " + file + "yolo.avi --yolo yolo-coco")


# read video and initialize matrices
cap = cv2.VideoCapture('videos/' + file + '.mp4') # read video
ret, frame1 = cap.read() # read initial frame
prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)   
hsv[...,1] = 255
writer = None # initialze writing to video

while(1):
    ret, frame2 = cap.read()
    next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])  # convert to polar coord
    hsv[...,0] = ang*180/np.pi/2                          		# set angle to hue 
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)  # set magnitue to value
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    # display in gui
    cv2.imshow('frame2',rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',rgb)
    prvs = next

    # write to video
    if writer is None:
    	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    	writer = cv2.VideoWriter(file + "optical.avi", fourcc, 30,(frame2.shape[1], frame2.shape[0]), True)
    # write the output frame to disk
    writer.write(rgb)
	
	
cap.release()
cv2.destroyAllWindows()
writer.release()

print("done")
