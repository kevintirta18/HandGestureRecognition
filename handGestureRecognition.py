import cv2
import numpy as np 


def standardize_image(img):
    img = cv2.resize(img, (200,200))
    return(img)

def subtract_background(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 10, 60], dtype = "uint8") 
    upper = np.array([30, 150, 255], dtype = "uint8")
    
    mask = cv2.inRange(img, lower, upper)
    img = cv2.bitwise_and(img, img, mask = mask)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
    kernel_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    kernel_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_7)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_5)
    img = cv2.morphologyEx(img, cv2.MORPH_DILATE, kernel_5)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel_3)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_5)
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel_5)
    return(img)

def find_palm_point(img):
    dist_mat = cv2.distanceTransform(img, cv2.DIST_L2, 5)

    result = np.where(dist_mat == np.amax(dist_mat))
    palm_points = list(zip(result[1], result[0]))
    return(palm_points[0])

def get_inner_circle(img, palm_point):
    img_inv = cv2.bitwise_not(img)
    is_found = 0
    d = 3
    radius = 1
    start_x = palm_point[0] - radius
    start_y = palm_point[1] - radius
    while(not is_found):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(d, d))
        test = img_inv[start_y-1:palm_point[1]+radius, start_x-1:palm_point[0]+radius]
        and_mat = cv2.bitwise_and(kernel, test)
        if (radius == 20):
            is_found = 1
        else:
            d = d+2
            radius = radius+1
            start_x = palm_point[0] - radius
            start_y = palm_point[1] - radius
    return(d)

def check_boundary(img, ref_x, ref_y):
    is_boundary = 0
    if ((img[ref_y-1, ref_x-1]) == 0 or 
            img[ref_y-1, ref_x] == 0 or
            img[ref_y-1, ref_x+1] == 0 or
            img[ref_y, ref_x+1] == 0 or
            img[ref_y+1, ref_x+1] == 0 or
            img[ref_y+1, ref_x] == 0 or
            img[ref_y+1, ref_x-1] == 0 or
            img[ref_y, ref_x-1] == 0 ):
        is_boundary = 1

    return(is_boundary)

def removeDuplicates(lst): 
    return list(set([i for i in lst])) 

def get_palm_mask_points(img, palm_point, d, first):
    d = int(1.2*d)
    mask_palm_points = []
    dist_two_consec_points = []
    dist_two_consec_points.append(0)
    last_max_dist = 0
    i = 0
    last_radius = 200
    
    for angle in range(20):
        #x, y of larger circle from palm_point
        x = palm_point[0] + int(d*np.cos(angle*18/180*np.pi))
        y = palm_point[1] + int(d*np.sin(angle*18/180*np.pi))
        last_radius = 200
        for theta in range(20):
            is_found = 0
            for radius in range(20):
                if(radius > last_radius):
                    break
                x_ = x + int(radius*np.cos(theta*18/180*np.pi))
                y_ = y + int(radius*np.sin(theta*18/180*np.pi))
                if (x_ > 199 or y_ > 199):
                    pass
                else:
                    if (img[y_, x_] == 0):
                        if (check_boundary(img, x_, y_)):
                            mask_palm_points.append((x_, y_)) 
                            i = i+1
                            distance = np.sqrt(((mask_palm_points[i-1][0]-mask_palm_points[i-2][0])**2+
                                                            (mask_palm_points[i-1][1]-mask_palm_points[i-2][1])**2))
                            dist_two_consec_points.append(distance)
                            if (distance > last_max_dist):
                                max_dist_endpoint_index = i-1
                                last_max_dist = dist_two_consec_points[i]
                            is_found = 1
                            last_radius = radius
                            break
    # Left to Right
    # This has bug
    if (mask_palm_points[max_dist_endpoint_index-1][0] < mask_palm_points[max_dist_endpoint_index][0]):
        wrist_points = ([mask_palm_points[max_dist_endpoint_index-1], mask_palm_points[max_dist_endpoint_index]])
    else:
        wrist_points = ([mask_palm_points[max_dist_endpoint_index], mask_palm_points[max_dist_endpoint_index-1]])
    mask_palm_points = removeDuplicates(mask_palm_points)
    return(mask_palm_points, wrist_points)

def rotate_hand(img, wrist_points, palm_points):
    middle_wrist = (wrist_points[0][0]+int((wrist_points[1][0]-wrist_points[0][0])/2), 
                    wrist_points[0][1]+int((wrist_points[1][1]-wrist_points[0][1])/2))

    rot_vector = (middle_wrist[0]-palm_points[0], middle_wrist[1]-palm_points[1])
    up_vector = (0, -1)
    angle = np.arccos((np.dot(rot_vector, up_vector))/(np.sqrt(rot_vector[0]**2+rot_vector[1]**2))/(np.sqrt(up_vector[0]**2+up_vector[1]**2)))
    degree = 180-angle*180/np.pi
    (h,w) = img.shape[:2]
    (cX, cY) = (w//2, h//2)
    if (middle_wrist[0] > palm_points[0]):
        rot_mat = cv2.getRotationMatrix2D((cX, cY), -degree, 1)
    else:
        rot_mat = cv2.getRotationMatrix2D((cX, cY), degree, 1)
    cos = np.abs(rot_mat[0, 0])
    sin = np.abs(rot_mat[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    rot_mat[0, 2] += (nW / 2) - cX
    rot_mat[1, 2] += (nH / 2) - cY

    img = cv2.warpAffine(img, rot_mat, (nW, nH))
 
    return(img)

def generate_mask_palm(img, mask_palm_points):
    temp = np.zeros_like(img)
    temp = cv2.fillPoly(temp, [np.asarray(mask_palm_points)], (255,255,255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,13))
    temp = cv2.morphologyEx(temp, cv2.MORPH_CLOSE, kernel)
    return(temp)

def segmentation(img, mask, palm_points, d):
    for i in range(int(palm_points[1]+0.7*d), img.shape[0]):
        mask[i, :] = 255
    return(cv2.bitwise_and(img, cv2.bitwise_not(mask)))

def detect_min_rectangle(img, val):
    threshold = val
    canny_output = cv2.Canny(img, threshold, threshold * 2)
    contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the rotated rectangles and ellipses for each contour
    minRect = [None]*len(contours)
    minEllipse = [None]*len(contours)
    for i, c in enumerate(contours):
        minRect[i] = cv2.minAreaRect(c)

    aspect_ratio_threshold = [0.8, 1.2]
    size_threshold = 20
    j = 0

    # Removing non-finger contour based on aspect ratio and size
    for i in range(len(minRect)):
        if (    ((minRect[i-j][1][1] / minRect[i-j][1][0] > aspect_ratio_threshold[0]) and 
                (minRect[i-j][1][1] / minRect[i-j][1][0] < aspect_ratio_threshold[1])) or
                (minRect[i-j][1][0] * minRect[i-j][1][1] < size_threshold)):
            del minRect[i-j]
            j = j+1
    return(minRect)

def thumb_detection(img, min_rect, palm_points, wrist_points):
    wrist_vector = (wrist_points[1][0] - wrist_points[0][0], wrist_points[1][1] - wrist_points[0][1])
    for i in range(len(min_rect)):
        candidate_vector = (min_rect[i][0][0] - palm_points[0], min_rect[i][0][1] - palm_points[1])
        angle = np.arccos((np.dot(candidate_vector, wrist_vector))/(np.sqrt(candidate_vector[0]**2+candidate_vector[1]**2))/(np.sqrt(wrist_vector[0]**2+wrist_vector[1]**2)))
        if (angle*np.pi/180 < 50):
            return (1, min_rect[i], i)    
    return (0, 0, -1)

def palm_line_detection(img, palm_points, wrist_points, d, has_thumb):
    wrist_vector = (wrist_points[1][0] - wrist_points[0][0], wrist_points[1][1] - wrist_points[0][1])
    layer2 = 0
    is_found = 0
    for i in range(wrist_points[0][1], 0, -1):
        count = 0
        if (is_found):
            break
        for j in range(1, img.shape[1]-1):
            if (int(img[i][j]) - int(img[i][j-1])) > 0:
                count = count+1
                if count > 2:
                    palm_lines_y = i
                    palm_lines_x = j
                    if (has_thumb == 0):
                        return(palm_lines_x, palm_lines_y)
                    else:
                        layer2 = 1
                        is_found = 1
                        
    if (layer2):
        if (palm_lines_x > img.shape[1]//2):
            for i in range(palm_lines_y, 0, -1):
                count = 0
                for j in range(1, palm_lines_x):
                    if (int(img[i][j]) - int(img[i][j-1])) > 0:
                        count = count+1
                        if count > 3:
                            palm_lines_y = i
                            palm_lines_x = j
                            return(palm_lines_x, palm_lines_y)
        else:
            for i in range(palm_lines_y, 0, -1):
                count = 0
                for j in range(palm_lines_x, img.shape[1]-1):
                    if (int(img[i][j]) - int(img[i][j-1])) > 0:
                        count = count+1
                        if count > 3:
                            palm_lines_y = i
                            palm_lines_x = j
                            return(palm_lines_x, palm_lines_y)
    return (0)

def finger_segmentation(img, min_rect_fingers, palm_lines_x, palm_lines_y, d, has_thumb):
    palm_line = [(palm_lines_x, palm_lines_y), (palm_lines_x-2*d, palm_lines_y)]
    palm_line = (palm_line[1], palm_line[0])
    typical_width = min_rect_fingers[1][0]
    # divider_points = [palm_lines_x-2*d, palm_lines_x-(2*2*d//3), palm_lines_x-(2*d//3)]
    divider_points = [palm_lines_x+2*d, palm_lines_x+(2*2*d//3), palm_lines_x+(2*d//3)]
    # Try to use info from fingers width

    detected_fingers_index = [None]*len(min_rect_fingers)
    # 0 = thumb
    for i in range(len(min_rect_fingers)):
        loc_x = int(1.1*min_rect_fingers[i][0][0])
        loc_y = min_rect_fingers[i][0][1]
        if(has_thumb):
            if (loc_y < palm_lines_y):
                if (loc_x < divider_points[2]):
                    detected_fingers_index[i] = 4
                elif (loc_x < divider_points[1]):
                    detected_fingers_index[i] = 3
                elif (loc_x < divider_points[0]):
                    detected_fingers_index[i] = 2
                else:
                    detected_fingers_index[i] = 1
            else:
                detected_fingers_index[i] = 0
        else:
            if (loc_x < divider_points[2]):
                detected_fingers_index[i] = 4
            elif (loc_x < divider_points[1]):
                detected_fingers_index[i] = 3
            elif (loc_x < divider_points[0]):
                detected_fingers_index[i] = 2
            else:
                detected_fingers_index[i] = 1
    return (detected_fingers_index)
    
def detect_hand_gesture(img):
    img = standardize_image(img)
    original = img
    img = subtract_background(img)
    palm_points_vec = find_palm_point(img)
    d = get_inner_circle(img, palm_points_vec)

    mask_palm, wrist_points = get_palm_mask_points(img, palm_points_vec, d, original)
    mask = generate_mask_palm(img, mask_palm)

    img = rotate_hand(img, wrist_points, palm_points_vec)
    original = rotate_hand(original, wrist_points, palm_points_vec)
    mask = rotate_hand(mask, wrist_points, palm_points_vec)
    
    new_palm_points_vec = find_palm_point(img)
    new_mask_palm, new_wrist_points = get_palm_mask_points(img, new_palm_points_vec, d, original)

    fingers = segmentation(img, mask, new_palm_points_vec, d)

    min_rect = detect_min_rectangle(fingers, 127)
    has_thumb, thumb, min_rect_index = thumb_detection(fingers, min_rect, new_palm_points_vec, new_wrist_points)
    palm_lines_x, palm_lines_y = palm_line_detection(fingers, new_palm_points_vec, new_wrist_points, d, has_thumb)

    detected_fingers = finger_segmentation(fingers, min_rect, palm_lines_x, palm_lines_y, d, has_thumb)

    # cv2.line(original, (int(new_palm_points_vec[0]- d), palm_lines_y), (int(new_palm_points_vec[0]+d), palm_lines_y), (255,0,0), 2)
    # cv2.circle(original, (new_palm_points_vec), 2, (255,0,0), 2)
    # cv2.circle(original, (new_wrist_points[0]), 2, (0,255,0), 2)
    # cv2.circle(original, (new_wrist_points[1]), 2, (0,255,0), 2)
    
    for i in range(len(min_rect)):
        box = cv2.boxPoints(min_rect[i])
        box = np.intp(box) #np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
        cv2.drawContours(original, [box], 0, (255,0,0))

    for i in range(len(min_rect)):
        cv2.putText(original, str(detected_fingers[i]), (int(min_rect[i][0][0]), int(min_rect[i][0][1])), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 2)
    return(original)


img = cv2.imread("test.jpg")
fingers = detect_hand_gesture(img)
cv2.imshow("fingers", fingers)
cv2.waitKey(0)