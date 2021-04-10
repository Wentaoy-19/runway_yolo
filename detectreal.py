import argparse
import os
import shutil
import time
from pathlib import Path
from PIL import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

import numpy as np
import math

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized


from detect import detect

def rotation(img):
    h, w = img.shape[:2]
    blured = cv2.blur(img,(5,5))    
    mask = np.zeros((h+2, w+2), np.uint8)  
    cv2.floodFill(blured, mask, (w-1,h-1), (255,255,255), (2,2,2),(3,3,3),8)
    cv2.floodFill(blured, mask, (0,0), (255,255,255), (2,2,2),(3,3,3),8)
    cv2.floodFill(blured, mask, (0,h-1), (255,255,255), (2,2,2),(3,3,3),8)
    cv2.floodFill(blured, mask, (w-1,0), (255,255,255), (2,2,2),(3,3,3),8)
    gray = cv2.cvtColor(blured,cv2.COLOR_BGR2GRAY) 
    ret, binary = cv2.threshold(gray,250,255,cv2.THRESH_BINARY) 
    image=[]
    for i in range(len(binary)):
        for j in range(len(binary[0])):
            if(binary[i][j]==0):
                image.append([j,i])
    image=np.float32(image)
    rect = cv2.minAreaRect(image)
    
    box = cv2.boxPoints(rect)

    print(box)

    
    
    left_point_x = np.min(box[:, 0])
    right_point_x = np.max(box[:, 0])
    top_point_y = np.min(box[:, 1])
    bottom_point_y = np.max(box[:, 1])

    left_point_y = box[:, 1][np.where(box[:, 0] == left_point_x)][0]
    right_point_y = box[:, 1][np.where(box[:, 0] == right_point_x)][0]
    top_point_x = box[:, 0][np.where(box[:, 1] == top_point_y)][0]
    bottom_point_x = box[:, 0][np.where(box[:, 1] == bottom_point_y)][0] 
#
    cv2.circle(img,(int(left_point_x),int(left_point_y)),3,(255,0,0),-1)
    cv2.circle(img,(int(right_point_x),int(right_point_y)),3,(255,0,0),-1)
    cv2.circle(img,(int(top_point_x),int(top_point_y)),3,(255,0,0),-1)
    cv2.circle(img,(int(bottom_point_x),int(bottom_point_y)),3,(255,0,0),-1)
    
    cv2.line(img,(int(left_point_x),int(left_point_y)),(int(top_point_x),int(top_point_y)),(255,0,0),2,8)
    cv2.line(img,(int(top_point_x),int(top_point_y)),(int(right_point_x),int(right_point_y)),(255,0,0),2,8) 
    cv2.line(img,(int(right_point_x),int(right_point_y)),(int(bottom_point_x),int(bottom_point_y)),(255,0,0),2,8)
    cv2.line(img,(int(bottom_point_x),int(bottom_point_y)),(int(left_point_x),int(left_point_y)),(255,0,0),2,8)
#  
    
    if((right_point_x-bottom_point_x)**2+(right_point_y-bottom_point_y)**2>=(right_point_x-top_point_x)**2+(right_point_y-top_point_y)**2):
        if(int(box[0][0])==int(box[1][0]) or int(box[0][1])==int(box[1][1])):
            angle1=90
        else:
            angle1=math.atan((bottom_point_y-right_point_y)/(right_point_x-bottom_point_x))/3.1415926535*180
    else:
        angle1=180+math.atan((right_point_y-top_point_y)/(top_point_x-right_point_x))/3.1415926535*180

    #angle=-rect[2]
    
    
    #cv2.imshow("binary",binary) 
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return angle1

###################################################################################
'''def rotation(img):
    h, w = img.shape[:2]
    blured = cv2.blur(img,(5,5))    
    mask = np.zeros((h+2, w+2), np.uint8)  
    cv2.floodFill(blured, mask, (w-1,h-1), (255,255,255), (2,2,2),(3,3,3),8)
    cv2.floodFill(blured, mask, (0,0), (255,255,255), (2,2,2),(3,3,3),8)
    cv2.floodFill(blured, mask, (0,h-1), (255,255,255), (2,2,2),(3,3,3),8)
    cv2.floodFill(blured, mask, (w-1,0), (255,255,255), (2,2,2),(3,3,3),8)
    gray = cv2.cvtColor(blured,cv2.COLOR_BGR2GRAY) 
    ret, binary = cv2.threshold(gray,250,255,cv2.THRESH_BINARY) 
    image=[]
    for i in range(len(binary)):
        for j in range(len(binary[0])):
            if(binary[i][j]==0):
                image.append([j,i])
    image=np.float32(image)
    rect = cv2.minAreaRect(image)
    angle=-rect[2]
    return angle
    #cv2.imshow("binary", binary) 
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()'''
################################################################################


def detectreal(model,input):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./tt1.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-dir', type=str, default='inference/output', help='directory to save results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()

    half = False
    device = 'cpu'
    imgsz = 640
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    dataset = LoadImages(input)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    img = torch.from_numpy(dataset.img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0    # 0 - 255 to 0.0 - 1.0
    

    returnvalue = []
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=opt.augment)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    if(pred[0]!=None):
        print(pred[0])
        for i in enumerate(pred[0]):
            ix =i[1]
            if(int(ix[5])==67):
                #print(ix[5])
                angle1=rotation(dataset.img0[int(ix[1]):int(ix[3]),int(ix[0]):int(ix[2])])
                print(angle1)
                ix=ix.numpy()
                returnvalue.append(angle1)
                returnvalue.append(int((ix[0]+ix[2])/2))
                returnvalue.append(int((ix[1]+ix[3])/2))
                #ix.append(angle1)
                cv2.circle(dataset.img0,(int((ix[0]+ix[2])/2),int((ix[1]+ix[3])/2)),5,(255,0,0),5)
                returnimg = Image.fromarray(np.uint8(dataset.img0))
                return np.array(returnvalue),returnimg
            else:
                continue
        return np.array([0,0]),Image.fromarray(np.uint8(dataset.img0))
    return np.array([0,0]),Image.fromarray(np.uint8(dataset.img0))


