import argparse
import os
import shutil
import time
from pathlib import Path

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


###################################################################################


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


def detect():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
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
    
    save_img = False
    out, source, weights, view_img, save_txt, imgsz = \
        opt.save_dir, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size

    device = select_device('cpu');

    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model

'''
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size


    
    vid_path, vid_writer = None, None

    dataset = LoadImages(input)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
    img = torch.from_numpy(dataset.img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0    # 0 - 255 to 0.0 - 1.0
    


    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=opt.augment)[0]
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    if(len(pred)>=1):
        for i in enumerate(pred[0]):
            print(i)
            return i[1].numpy()
    

    
    for i in pred[0]:
        if(int(i[5])==65):
            print(i.numpy())
            return i.numpy()
'''
    ##return np.array([1,2])

'''
    dangle=0
    angle=0
    angle1=0

    for path, img, im0s, vid_cap in dataset:

        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)


##############################################################          
        #print("one frame")
        #print("current time:", (time.time()-t0))
        if(pred[0]!=None):                       
            for ix in pred[0]:
                if(ix[5]==67.0):
                    print([float((ix[2]+ix[0])/2),float((ix[3]+ix[1])/2)])
                    
                    break
##############################################################


        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            txt_path = str(Path(out) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
   
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()


                         #####################################################################################################
                if(ix[5]==67.0): 
                    angle1=rotation(im0[int(ix[1]):int(ix[3]),int(ix[0]):int(ix[2])])   
                    dangle=angle1-angle
                    if(dangle>70 or dangle<-70):
                        dangle=0
                    angle=angle1
                    print(dangle,angle1)  

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, conf, *xywh) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line) + '\n') % line)

                    if save_img or view_img:  # Add bbox to image
                        if(names[int(cls)]=="cell phone"):                  ########################################################################
                            label = '%s %.2f' % (names[int(cls)], angle1)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            #print(int(xyxy[2]),int(xyxy[3]))
                            break


 
                #imgcv=im0.detach().numpy()
                if(ix[5]==67.0):
                    cv2.circle(im0,(int((ix[0]+ix[2])/2),int((ix[1]+ix[3])/2)),5,(255,0,0),5)

                #cv2.circle(im0,(int((xyxy[0]+xyxy[2])/2),int((xyxy[1]+xyxy[3])/2)),5,(255,0,0),5)
            # Print time (inference + NMS)
            #print('%sDone. (%.3fs)' % (s, t2 - t1))
            

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)###############################################3
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fourcc = 'mp4v'  # output video codec
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % Path(out))

    #print('Done. (%.3fs)' % (time.time() - t0))

'''

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./tt1.jpg', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
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
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
'''