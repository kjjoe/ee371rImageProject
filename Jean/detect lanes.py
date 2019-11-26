import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import time

from moviepy.editor import VideoFileClip
from IPython.display import HTML

##########   Calibration    ############

def draw_imgs(lst, rows, cols=2, figsize=(10, 25), dosave= False, save_dir=""):
    assert(len(lst) > 0)
    assert(rows > 0)
    if dosave:
        assert(os.path.exists(save_dir))
    fig = plt.figure(figsize=figsize)
    fig.tight_layout()
    for i in range(1, rows * cols +1):
        fig.add_subplot(rows, cols, i)
        img = mpimg.imread(CAL_IMGS + "/"+calib_files[i-1])
        plt.imshow(img)
    plt.show()
    if dosave:
        fig.savefig(save_dir + "/op_" + str(time.time()) + ".jpg")
        
        
def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

########## END   Calibration    ############
        
########## Distortion correction  ############

def undistort(img_name, objpoints, imgpoints):
    img = cv2.imread(img_name)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist= cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def undistort_no_read(img, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[1:], None, None)
    undist= cv2.undistort(img, mtx, dist, None, mtx)
    return undist


##########################################        


########## Gradient and Color transform  ############

def abs_thresh(img, sobel_kernel=3, mag_thresh=(0,255), return_grad= False, direction ='x'):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad = None
    scaled_sobel = None
    
    # Sobel x
    if direction.lower() == 'x':
        grad = cv2.Sobel(gray, cv2.CV_64F, 1, 0,ksize=sobel_kernel) # Take the derivative in x       
    # Sobel y
    else:
        grad = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
        
    if return_grad == True:
        return grad
        
    abs_sobel = np.absolute(grad) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel < mag_thresh[1])] = 1
    
    return grad_binary
        

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):    
    xgrad =  abs_thresh(img, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh, return_grad=True)
    ygrad =  abs_thresh(img, sobel_kernel=sobel_kernel, mag_thresh=mag_thresh, return_grad=True, direction='y')
    
    magnitude = np.sqrt(np.square(xgrad)+np.square(ygrad))
    abs_magnitude = np.absolute(magnitude)
    scaled_magnitude = np.uint8(255*abs_magnitude/np.max(abs_magnitude))
    mag_binary = np.zeros_like(scaled_magnitude)
    mag_binary[(scaled_magnitude >= mag_thresh[0]) & (scaled_magnitude < mag_thresh[1])] = 1
    
    return mag_binary


def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    xgrad =  cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    ygrad =  cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    xabs = np.absolute(xgrad)
    yabs = np.absolute(ygrad)
    
    grad_dir = np.arctan2(yabs, xabs)
    
    binary_output = np.zeros_like(grad_dir).astype(np.uint8)
    binary_output[(grad_dir >= thresh[0]) & (grad_dir < thresh[1])] = 1
    return binary_output


def get_rgb_thresh_img(img, channel='R', thresh=(0, 255)):
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if channel == 'R':
        bin_img = img1[:, :, 0]
    if channel == 'G' :
        bin_img = img1[:, :, 1]
    if channel == 'B' :
        bin_img = img1[:, :, 2]
        
    binary_img = np.zeros_like(bin_img).astype(np.uint8) 
    binary_img[(bin_img >= thresh[0]) & (bin_img < thresh[1])] = 1
    
    return binary_img


def get_hls_lthresh_img(img, thresh=(0, 255)):
    hls_img= cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    L = hls_img[:, :, 1]

    binary_output = np.zeros_like(L).astype(np.uint8)    
    binary_output[(L >= thresh[0]) & (L < thresh[1])] = 1
    
    return binary_output


def get_hls_sthresh_img(img, thresh=(0, 255)):
    hls_img= cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    S = hls_img[:, :, 2]

    binary_output = np.zeros_like(S).astype(np.uint8)    
    binary_output[(S >= thresh[0]) & (S < thresh[1])] = 1
    
    return binary_output


def get_lab_athresh_img(img, thresh=(0,255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    A = lab_img[:, :, 1]
    
    bin_op = np.zeros_like(A).astype(np.uint8)
    bin_op[(A >= thresh[0]) & (A < thresh[1])] = 1
    
    return bin_op


def get_lab_bthresh_img(img, thresh=(0,255)):
    lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    B = lab_img[:, :, 2]
    
    bin_op = np.zeros_like(B).astype(np.uint8)
    bin_op[(B >= thresh[0]) & (B < thresh[1])] = 1
    
    return bin_op


def get_bin_img(img, kernel_size=3, sobel_dirn='X', sobel_thresh=(0,255), r_thresh=(0, 255), 
                s_thresh=(0,255), b_thresh=(0, 255), g_thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float32)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      
    if sobel_dirn == 'X':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = kernel_size)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = kernel_size)
        
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    sbinary = np.zeros_like(scaled_sobel)
    sbinary[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    
    combined = np.zeros_like(sbinary)
    combined[(sbinary == 1)] = 1

    # Threshold R color channel
    r_binary = get_rgb_thresh_img(img, thresh= r_thresh)
    
    # Threshhold G color channel
    g_binary = get_rgb_thresh_img(img, thresh= g_thresh, channel='G')
    
    # Threshhold B in LAB
    b_binary = get_lab_bthresh_img(img, thresh=b_thresh)
    
    # Threshold color channel
    s_binary = get_hls_sthresh_img(img, thresh=s_thresh)

    # If two of the three are activated, activate in the binary image
    combined_binary = np.zeros_like(combined)
    combined_binary[(r_binary == 1) | (combined == 1) | (s_binary == 1)| (b_binary == 1) | (g_binary == 1)] = 1

    return combined_binary
#######################################
    
#######  Perspective Transform  #########
    
def transform_image(img, offset=250, src=None, dst=None):    
    img_size = (img.shape[1], img.shape[0])
    
    out_img_orig = np.copy(img)
       
    leftupper  = (585, 460)
    rightupper = (705, 460)
    leftlower  = (210, img.shape[0])
    rightlower = (1080, img.shape[0])
    
    
    warped_leftupper = (offset,0)
    warped_rightupper = (offset, img.shape[0])
    warped_leftlower = (img.shape[1] - offset, 0)
    warped_rightlower = (img.shape[1] - offset, img.shape[0])
    
    color_r = [0, 0, 255]
    color_g = [0, 255, 0]
    line_width = 5
    
    if src is not None:
        src = src
    else:
        src = np.float32([leftupper, leftlower, rightupper, rightlower])
        
    if dst is not None:
        dst = dst
    else:
        dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])
    
    cv2.line(out_img_orig, leftlower, leftupper, color_r, line_width)
    cv2.line(out_img_orig, leftlower, rightlower, color_r , line_width * 2)
    cv2.line(out_img_orig, rightupper, rightlower, color_r, line_width)
    cv2.line(out_img_orig, rightupper, leftupper, color_g, line_width)
    
    # calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)
    
    # Warp the image
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
    out_warped_img = np.copy(warped)
    
    cv2.line(out_warped_img, warped_rightupper, warped_leftupper, color_r, line_width)
    cv2.line(out_warped_img, warped_rightupper, warped_rightlower, color_r , line_width * 2)
    cv2.line(out_warped_img, warped_leftlower, warped_rightlower, color_r, line_width)
    cv2.line(out_warped_img, warped_leftlower, warped_leftupper, color_g, line_width)
    
    return warped, M, minv, out_img_orig, out_warped_img

#########################################


###########  Lane line pixel detection and polynomial fitting  #######
    
def find_lines(warped_img, nwindows=9, margin=80, minpix=40):
    
    # Take a histogram of the bottom half of the image
    histogram = np.sum(warped_img[warped_img.shape[0]//2:,:], axis=0)
        
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped_img, warped_img, warped_img)) * 255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped_img.shape[0]//nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped_img.shape[0] - (window+1)*window_height
        win_y_high = warped_img.shape[0] - window*window_height
        
        ### Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  
        win_xleft_high = leftx_current + margin  
        win_xright_low =  rightx_current - margin 
        win_xright_high = rightx_current + margin  
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low), (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low), (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) &\
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low ) & (nonzeroy < win_y_high) &\
                            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img

def fit_polynomial(binary_warped, nwindows=9, margin=100, minpix=50, show=True):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, left_lane_inds, right_lane_inds, out_img \
        = find_lines(binary_warped, nwindows=nwindows, margin=margin, minpix=minpix)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    if show == True:
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    return left_fit, right_fit, left_fitx, right_fitx, left_lane_inds, right_lane_inds, out_img
######################################################################



###########   Skip the sliding windows step once you've found the lines  #########
    
def search_around_poly(binary_warped, left_fit, right_fit, ymtr_per_pixel, xmtr_per_pixel, margin=80):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit second order polynomial to for for points on real world   
    left_lane_indices = np.polyfit(lefty*ymtr_per_pixel, leftx*xmtr_per_pixel, 2)
    right_lane_indices = np.polyfit(righty*ymtr_per_pixel, rightx*xmtr_per_pixel, 2)
    
    return left_fit, right_fit, left_lane_indices, right_lane_indices

#########################################################



##########   Radius of curvature  ################
    
def radius_curvature(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    y_eval = np.max(ploty)
    
    left_fit_cr = np.polyfit(ploty*ymtr_per_pixel, left_fitx*xmtr_per_pixel, 2)
    right_fit_cr = np.polyfit(ploty*ymtr_per_pixel, right_fitx*xmtr_per_pixel, 2)
    
    # find radii of curvature
    left_rad = ((1 + (2*left_fit_cr[0]*y_eval*ymtr_per_pixel + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_rad = ((1 + (2*right_fit_cr[0]*y_eval*ymtr_per_pixel + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    return (left_rad, right_rad)


def dist_from_center(img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel):
    ## Image mid horizontal position 
    #xmax = img.shape[1]*xmtr_per_pixel
    ymax = img.shape[0]*ymtr_per_pixel
    
    center = img.shape[1] / 2
    
    lineLeft = left_fit[0]*ymax**2 + left_fit[1]*ymax + left_fit[2]
    lineRight = right_fit[0]*ymax**2 + right_fit[1]*ymax + right_fit[2]
    
    mid = lineLeft + (lineRight - lineLeft)/2
    dist = (mid - center) * xmtr_per_pixel
    if dist >= 0. :
        message = 'Vehicle location: {:.2f} m right'.format(dist)
    else:
        message = 'Vehicle location: {:.2f} m left'.format(abs(dist))
    
    return message


def draw_lines(img, left_fit, right_fit, minv):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_warp = np.zeros_like(img).astype(np.uint8)
    
    # Find left and right points.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix 
    unwarp_img = cv2.warpPerspective(color_warp, minv, (img.shape[1], img.shape[0]), flags=cv2.WARP_FILL_OUTLIERS+cv2.INTER_CUBIC)
    return cv2.addWeighted(img, 1, unwarp_img, 0.3, 0)


def show_curvatures(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel):
    (left_curvature, right_curvature) = radius_curvature(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    dist_txt = dist_from_center(img, leftx, rightx, xmtr_per_pixel, ymtr_per_pixel)
    
    out_img = np.copy(img)
    avg_rad = round(np.mean([left_curvature, right_curvature]),0)
    cv2.putText(out_img, 'Average lane curvature: {:.2f} m'.format(avg_rad), 
                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    cv2.putText(out_img, dist_txt, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    
    return out_img

#############################################




#############  Pipeline for video   #############
    
class Lane():
    def __init__(self, max_counter):
        self.current_fit_left=None
        self.best_fit_left = None
        self.history_left = [np.array([False])] 
        self.current_fit_right=None
        self.best_fit_right = None
        self.history_right = [np.array([False])] 
        self.counter = 0
        self.max_counter = 1
        self.src = None
        self.dst = None
        
    def set_presp_indices(self, src, dest):
        self.src = src
        self.dst = dest
        
    def reset(self):
        self.current_fit_left=None
        self.best_fit_left = None
        self.history_left =[np.array([False])] 
        self.current_fit_right = None
        self.best_fit_right = None
        self.history_right =[np.array([False])] 
        self.counter = 0
        
    def update_fit(self, left_fit, right_fit):
        if self.counter > self.max_counter:
            self.reset()
        else:
            self.current_fit_left = left_fit
            self.current_fit_right = right_fit
            self.history_left.append(left_fit)
            self.history_right.append(right_fit)
            self.history_left = self.history_left[-self.max_counter:] if len(self.history_left) > self.max_counter else self.history_left
            self.history_right = self.history_right[-self.max_counter:] if len(self.history_right) > self.max_counter else self.history_right
            self.best_fit_left = np.mean(self.history_left, axis=0)
            self.best_fit_right = np.mean(self.history_right, axis=0)
        
    def process_image(self, image):
        img = undistort_no_read(image, objpoints, imgpoints)
        
        combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh,
                                      r_thresh=r_thresh, s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
    
        if self.src is not None or self.dst is not None:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, src=self.src, dst= self.dst)
        else:
            warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
    
        xmtr_per_pixel=3.7/800
        ymtr_per_pixel=30/720
    
        if self.best_fit_left is None and self.best_fit_right is None:
            left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped, nwindows=15, show=False)
        else:
            left_fit, right_fit, left_lane_indices, right_lane_indices= search_around_poly(warped, self.best_fit_left, self.best_fit_right, xmtr_per_pixel, ymtr_per_pixel)
            
        self.counter += 1
        
        lane_img = draw_lines(img, left_fit, right_fit, unwarp_matrix)
        out_img = show_curvatures(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)
        
        self.update_fit(left_fit, right_fit)
        
        return out_img
################################################




if __name__ == "__main__":

    CAL_IMGS = "camera_cal"
    calib_files = os.listdir(CAL_IMGS)
    assert(len(calib_files) > 0)
    
    # Create directory to save output directory
    OUTDIR = "output_images/"
    create_dir(OUTDIR)
    # Just checking the image
    draw_imgs(calib_files, len(calib_files)//2, dosave=True, save_dir=OUTDIR)
    
    
    ##########   Calibration    ############
    nx = 9
    ny = 6
    
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:,:2] = np.mgrid[:nx, :ny].T.reshape(-1, 2)
    
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    failed =[]
    
    for idx, name in enumerate(calib_files):
        img = cv2.imread(CAL_IMGS + "/"+ name)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        if ret == True:
            objpoints.append(objp)
            
            corners = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            imgpoints.append(corners)
            
            # Draw and display the corners
#            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
#            f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
#            f.tight_layout()
#            ax1.imshow(cv2.cvtColor(cv2.imread(CAL_IMGS + "/"+ name), cv2.COLOR_BGR2RGB))
#            ax1.set_title("Original:: " + name, fontsize=18)
#            ax2.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
#            ax2.set_title("Corners:: "+ name, fontsize=18)
#            f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
            
        else:
            failed.append(name)
            
    print("Failed for images: [")
    print(failed)
    print("]")
#        
#    ##########  END Calibration    ############
#    
#    ########## Distortion correction  ##########
#    
#    undist = undistort(CAL_IMGS+"/calibration10.jpg", objpoints, imgpoints)
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
#    f.tight_layout()
#    ax1.imshow(cv2.cvtColor(cv2.imread(CAL_IMGS+"/calibration10.jpg"), cv2.COLOR_BGR2RGB))
#    ax1.set_title("Original:: calibration10.jpg" , fontsize=18)
#    ax2.imshow(cv2.cvtColor(undist,cv2.COLOR_BGR2RGB))
#    ax2.set_title("Undistorted:: calibration10.jpg", fontsize=18)
#    f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
#
#    images = glob.glob('test_images/test*.jpg')
#    for image in images:
#        undist = undistort(image, objpoints, imgpoints)
#        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
#        f.tight_layout()
#        ax1.imshow(cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB))
#        ax1.set_title("Original:: " + image , fontsize=18)
#        ax2.imshow(cv2.cvtColor(undist, cv2.COLOR_BGR2RGB))
#        ax2.set_title("Undistorted:: "+ image, fontsize=18)
#        f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
#
#    #########################################
#    
#    ########## Gradient and Color transform  ############
#    img = undistort(images[0], objpoints, imgpoints)
#    
#    combined_binary = abs_thresh(img, sobel_kernel=3, mag_thresh=(30, 100), direction='x')
#    warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary, offset=300)
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
#    f.tight_layout()
#    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#    ax1.set_title("Original:: " + image , fontsize=18)
#    ax2.imshow(warped, cmap='gray')
#    ax2.set_title("Transformed:: "+ image, fontsize=18)
#        
#    # Testing the threshholding
    kernel_size = 5
    mag_thresh = (30, 100)
    r_thresh = (235, 255)
    s_thresh = (165, 255)
    b_thresh = (160, 255)
    g_thresh = (210, 255)
#    
#    for image_name in images:
#        img = undistort(image_name, objpoints, imgpoints)
#        
#        combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh, 
#                                      s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
#        
#        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
#        f.tight_layout()
#        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#        ax1.set_title("Original:: " + image , fontsize=18)
#        ax2.imshow(combined_binary, cmap='gray')
#        ax2.set_title("Threshold Binary:: "+ image, fontsize=18)
#        f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")    
#    
#    #########################################
#    
#    
#    
#    ##############   Perspective transform   #####################
#    for image in images:
#        img = cv2.imread(image)
#        warped, M, minv, out_img_orig, out_warped_img = transform_image(img)
#        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
#        f.tight_layout()
#        ax1.imshow(cv2.cvtColor(out_img_orig, cv2.COLOR_BGR2RGB))
#        ax1.set_title("Original:: " + image , fontsize=18)
#        ax2.imshow(cv2.cvtColor(out_warped_img, cv2.COLOR_BGR2RGB))
#        ax2.set_title("Warped:: "+ image, fontsize=18)
#        f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
#    
#    
#    ########################################
#    
#
#    ############   Radius of curvature   ####################
#    
#    for image in images:    
#        img = undistort(image, objpoints, imgpoints)
#        
#        combined_binary = get_bin_img(img, kernel_size=kernel_size, sobel_thresh=mag_thresh, r_thresh=r_thresh, 
#                                      s_thresh=s_thresh, b_thresh = b_thresh, g_thresh=g_thresh)
#        warped, warp_matrix, unwarp_matrix, out_img_orig, out_warped_img = transform_image(combined_binary)
#        
#        xmtr_per_pixel=3.7/800
#        ymtr_per_pixel=30/720
#        
#        left_fit, right_fit, left_fitx, right_fitx, left_lane_indices, right_lane_indices, out_img = fit_polynomial(warped, nwindows=12, show=False)
#        lane_img = draw_lines(img, left_fit, right_fit, minv)
#        out_img = show_curvatures(lane_img, left_fit, right_fit, xmtr_per_pixel, ymtr_per_pixel)
#        
#        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,8))
#        f.tight_layout()
#        ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#        ax1.set_title("Original:: " + image , fontsize=18)
#        ax2.imshow(cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB))
#        ax2.set_title("Lane:: "+ image, fontsize=18)
#        f.savefig(OUTDIR + "/op_" + str(time.time()) + ".jpg")
#    
#    #########################################################
#    
#    
    #############   Pipeline for video   ##################
    clip1 = VideoFileClip("33.mp4")
    img = clip1.get_frame(0)
    
    leftupper  = (585, 460)
    rightupper = (705, 460)
    leftlower  = (210, img.shape[0])
    rightlower = (1080, img.shape[0])
        
    color_r = [255, 0, 0]
    color_g = [0, 255, 0]
    line_width = 5
        
    src = np.float32([leftupper, leftlower, rightupper, rightlower])
    
    cv2.line(img, leftlower, leftupper, color_r, line_width)
    cv2.line(img, leftlower, rightlower, color_r , line_width * 2)
    cv2.line(img, rightupper, rightlower, color_r, line_width)
    cv2.line(img, rightupper, leftupper, color_g, line_width)
    
    plt.imshow(img)


    lane1 = Lane(max_counter=5)
    
    leftupper  = (585, 460)
    rightupper = (705, 460)
    leftlower  = (210, img.shape[0])
    rightlower = (1080, img.shape[0])
        
    warped_leftupper = (250,0)
    warped_rightupper = (250, img.shape[0])
    warped_leftlower = (1050, 0)
    warped_rightlower = (1050, img.shape[0])
    
    src = np.float32([leftupper, leftlower, rightupper, rightlower])
    dst = np.float32([warped_leftupper, warped_rightupper, warped_leftlower, warped_rightlower])
    
    lane1.set_presp_indices(src, dst)
    
    output = "test_videos_output/project.mp4"
    clip1 = VideoFileClip("33.mp4")
    #clip1.reader.close()
    #clip1.audio.reader.close_proc()
    #clip1.audio.reader.close_proc()
    white_clip = clip1.fl_image(lane1.process_image)  
    #clip1.close()
    white_clip.preview(fps=25,audio=False) # don't generate/play the audio.
    ######################################################
    