                #A.U.T.O#
#Automatic Utility for Translating Objects#
from numpy import ones,vstack
from numpy.linalg import lstsq
from getkeys import key_check
from statistics import mean
import numpy as np
import cv2
import time
from PIL import ImageGrab
from directkeys import ReleaseKey, PressKey, W, A, S, D
import os

def keys_output(keys):
            #[A,W,D]#
    output = [0,0,0]

    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:
        output[1] = 1
    
    return output

file_name = 'training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data')
    np_load_old = np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
    training_data = list(np.load(file_name))
else:
    print('File does not exist, creating new')
    training_data = []

def draw_lines(img, lines):
    try:
        ys = []  
        for i in lines:
            for ii in i:
                ys += [ii[1],ii[3]]
        min_y = min(ys)
        max_y = 600
        new_lines = []
        line_dict = {}

        for idx,i in enumerate(lines):
            for xyxy in i:
                x_coords = (xyxy[0],xyxy[2])
                y_coords = (xyxy[1],xyxy[3])
                A = vstack([x_coords,ones(len(x_coords))]).T
                m, b = lstsq(A, y_coords)[0]

                x1 = (min_y-b) / m
                x2 = (max_y-b) / m

                line_dict[idx] = [m,b,[int(x1), min_y, int(x2), max_y]]
                new_lines.append([int(x1), min_y, int(x2), max_y])

        final_lanes = {}

        for idx in line_dict:
            final_lanes_copy = final_lanes.copy()
            m = line_dict[idx][0]
            b = line_dict[idx][1]
            line = line_dict[idx][2]
            
            if len(final_lanes) == 0:
                final_lanes[m] = [ [m,b,line] ]
                
            else:
                found_copy = False

                for other_ms in final_lanes_copy:

                    if not found_copy:
                        if abs(other_ms*1.2) > abs(m) > abs(other_ms*0.1):
                            if abs(final_lanes_copy[other_ms][0][1]*1.2) > abs(b) > abs(final_lanes_copy[other_ms][0][1]*0.8):
                                final_lanes[other_ms].append([m,b,line])
                                found_copy = True
                                break
                        else:
                            final_lanes[m] = [ [m,b,line] ]

        line_counter = {}

        for lanes in final_lanes:
            line_counter[lanes] = len(final_lanes[lanes])

        top_lanes = sorted(line_counter.items(), key=lambda item: item[1])[::-1][:2]

        lane1_id = top_lanes[0][0]
        lane2_id = top_lanes[1][0]

        def average_lane(lane_data):
            x1s = []
            y1s = []
            x2s = []
            y2s = []
            for data in lane_data:
                x1s.append(data[2][0])
                y1s.append(data[2][1])
                x2s.append(data[2][2])
                y2s.append(data[2][3])
            return int(mean(x1s)), int(mean(y1s)), int(mean(x2s)), int(mean(y2s)) 

        l1_x1, l1_y1, l1_x2, l1_y2 = average_lane(final_lanes[lane1_id])
        l2_x1, l2_y1, l2_x2, l2_y2 = average_lane(final_lanes[lane2_id])

        return [l1_x1, l1_y1, l1_x2, l1_y2], [l2_x1, l2_y1, l2_x2, l2_y2], lane1_id, lane2_id
    except:
         pass
def roi(img, verticies) :
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, verticies, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked



def process_img(image):
    original_img = image
    processed_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    processed_img = cv2.Canny(original_img, 200, 300)
    processed_img = cv2.GaussianBlur(processed_img, (5,5), 0)

    verti = np.array([[10,500],[10,300],[300,200],[500,200],[800,300],[800,500]])
    processed_img = roi(processed_img, [verti])

    lines = cv2.HoughLinesP(processed_img, 1, np.pi/180, 180, np.array([]), 100, 0)
    m1 = 0
    m2 = 0
    try:
        l1, l2, m1, m2 = draw_lines(original_img,lines)
        cv2.line(original_img, (l1[0], l1[1]), (l1[2], l1[3]), [0,255,0], 30)
        cv2.line(original_img, (l2[0], l2[1]), (l2[2], l2[3]), [0,255,0], 30)
    except:

        pass
    try:
        for coords in lines:
            coords = coords[0]
            try:
                cv2.line(processed_img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 3)
                
                
            except:
                pass
    except:
        pass

    return processed_img, original_img, m1, m2

def main():
    
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)
#    last_time = time.time()
    
#def straight():
#    PressKey(W)
#    ReleaseKey(A)
#    ReleaseKey(D)
#def left():
#    PressKey(A)
#    ReleaseKey(W)
#    ReleaseKey(D)
#
#def right():
#    PressKey(D)
#    ReleaseKey(A)
#    ReleaseKey(W)
#
#def slow():
#    ReleaseKey(W)
#    ReleaseKey(A)
#    ReleaseKey(D)

for i in list(range(4))[::-1]:
    print(i+1)
    time.sleep(1)


while True:
#        PressKey(W)
#        time.sleep(3)
#        ReleaseKey(W)

        img =  np.array(ImageGrab.grab(bbox=(0,30,800,630)))    
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (80,60))
        keys = key_check()
        output = keys_output(keys)
        training_data.append([img, output])

   
    #    print('Frame took {} seconds'.format(time.time()-last_time ))
    #    last_time = time.time()

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)


#        new_screen, original_img, m1, m2 = process_img(img)
#       # cv2.imshow('GrabLaneBackstage', new_screen)
#        cv2.imshow('FrontendLaneDetection',cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
#        #cv2.imshow('GrabLane',cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#        
#    #        
#    #    if m1 > 0 and m2 >0:
#    #        left()
#    #    if m1 < 0 and m2 <0:
#    #        right()
#    #    else:
#    #        straight()
#    #    
#        
#        if cv2.waitKey(25) & 0xFF == ord('q'):
#            cv2.destroyAllWindows()
#            break#