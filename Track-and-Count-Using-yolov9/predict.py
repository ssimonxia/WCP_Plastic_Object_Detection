import cv2
import pandas as pd
from torch import NoneType
from ultralytics import YOLO
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import imageio
from os import makedirs, remove
from datetime import datetime
from numpy import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt 
from config import Config

config = Config()

model=YOLO(config.model)

deepsort = None

object_counter = {}

object_counter1 = {}

line = config.line

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

# class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
# class_list = ["Aluminum", "Books", "Cardboard", "Glass", "HDPE", "LDPE", "OP", "PET", "PP", "PS", "RP"]
class_list = config.class_list

source = config.source

data_deque = {}

def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    
    deepsort = DeepSort('./deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
    # deepsort = DeepSort(model_path='./deep_sort/deep/checkpoint/ckpt.t7', max_age=70)
    

# label different classes with different color
def compute_color_for_labels(label):
    if label == 0: #person
        color = (85,45,255)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

def UI_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]

        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)

        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str
def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    # remove tracked point from buffer if object is lost
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # code to find center of bottom edge
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID of object
        id = int(identities[i]) if identities is not None else 0

        # create new buffer for new object
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # add center to buffer
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
          direction = get_direction(data_deque[id][0], data_deque[id][1])
          if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
              cv2.line(img, line[0], line[1], (255, 255, 255), 3)
              if "South" in direction:
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
              if "North" in direction:
                if obj_name not in object_counter1:
                    object_counter1[obj_name] = 1
                else:
                    object_counter1[obj_name] += 1
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # draw trail
        for i in range(1, len(data_deque[id])):
            # check if on buffer value is none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # generate dynamic thickness of trails
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # draw trails
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    
    #4. Display Count in top right corner
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = str(key) + ":" +str(value)
            cv2.line(img, (width - 500,25), (width,25), [85,45,255], 40)
            cv2.putText(img, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
            cv2.line(img, (width - 250, 65 + (idx*40)), (width, 65 + (idx*40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str, (width - 250, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 2, lineType = cv2.LINE_AA)

        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" +str(value)
            cv2.line(img, (20,25), (500,25), [85,45,255], 40)
            cv2.putText(img, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)    
            cv2.line(img, (20,65+ (idx*40)), (200,65+ (idx*40)), [85,45,255], 30)
            cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
    
    
    return img


def main():
    init_tracker()
    # tracker=Tracker()
    count=0
    cap=cv2.VideoCapture(source)

    # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = config.output_video_width
    frame_height = config.output_video_height
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    down={}
    counter_down=set()
    flag = 0
    images = []
    count_images = []
    output_video = cv2.VideoWriter("outputs/output_video.avi", cv2.VideoWriter_fourcc(*'MJPG'), fps, (frame_width, frame_height))
    while True:    
        
        ret,frame = cap.read()
        
        if frame is None:
            break
        
        frame = cv2.resize(frame, (frame_width, frame_height))
        
        original_frame = frame.copy()
        
        if not ret:
            break
        count += 1
        results=model.predict(frame)
        
        ##############################################33
        
        class_name = []
        
        for result in results:
            boxes = result.boxes # all information about bounding box
            probs = result.probs # probability, sometimes is None
            cls = boxes.cls.tolist() # Contain class indices
            xyxy = boxes.xyxy # x1, y1, x2, y2 of the bounding box
            conf = boxes.conf # confidence
            xywh = boxes.xywh # xc, yc, w, h
            for index in cls:
                class_name.append(class_list[int(index)])

        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float)
        class_name = np.array(class_name)

        tracks=deepsort.update(bboxes_xywh, conf, pred_cls, original_frame)
        
        
        if len(tracks[0]) > 0:
            bbox_xyxy = tracks[0][:, :4]
            identities = tracks[0][:, -1]
            object_id = tracks[0][:, -2]
            
            out_img = draw_boxes(frame, bbox_xyxy, class_list, object_id, identities)
            if config.visualize:
                cv2.imshow("frame", out_img)
            output_video.write(out_img)

            
            # cv2.imwrite(f"outputs/frames_{flag}.png", frame)
            # images.append(imageio.imread(f"outputs/frames_{flag}.png"))
            # remove(f"outputs/frames_{flag}.png")
            
            # count_label = set(object_counter.keys())
            # count_label1 = set(object_counter1.keys())
            # count_label = count_label.union(count_label1)
            # count_value = {}
            # #print(count_label, object_counter, object_counter1)
            # for ind in count_label:
            #     i = str(ind)
            #     if i in object_counter:
            #         if i in count_value:
            #             count_value[i] += object_counter[i]
            #         else:
            #             count_value[i] = object_counter[i]
                        
            #     if i in object_counter1:
            #         if i in count_value:
            #             count_value[i] += object_counter1[i]
            #         else:
            #             count_value[i] = object_counter1[i]
                    
            # plt.bar(list(count_label), list(count_value.values()), color = "maroon")
            # plt.xlabel("Class name")
            # plt.ylabel("Count")
            # plt.tight_layout()
            # plt.savefig(f"count_value_{flag}.png")
            # plt.show()
            # plt.close("all")
            # count_images.append(imageio.imread(f"count_value_{flag}.png"))
            # # count_tmp_img = cv2.imread(f"count_value_{flag}.png")
            # # cv2.imshow("count", count_tmp_img)
            # remove(f"count_value_{flag}.png")
        
        # for track in deepsort.tracker.tracks:
        # if len(tracks[0]) > 0:
        #     for track in tracks[0]:
        #         x3,y3,x4,y4 = track[:4]
        #         id = track[-2]
        #         object_id = track[-1]
        #         cx=int(x3+x4)//2
        #         cy=int(y3+y4)//2
        #         cv2.circle(frame,(cx,cy),4,(0,0,255),-1) #draw ceter points of bounding box
        #         cv2.rectangle(frame, (int(x3), int(y3)), (int(x4), int(y4)), (0, 255, 0), 2)  # Draw bounding box
        #         cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
                
        #         y = 308
        #         offset = 7
        
        #         ''' condition for red line '''
        #         if y < (cy + offset) and y > (cy - offset):
        #             ''' this if condition is putting the id and the circle on the object when the center of the object touched the red line.'''
            
        #             down[id]=cy   #cy is current position. saving the ids of the cars which are touching the red line first. 
        #             #This will tell us the travelling direction of the car.
        #         if id in down:         
        #             cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
        #             cv2.putText(frame,str(id),(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        #             counter_down.add(id) 

        # # line
        # text_color = (255,255,255)  # white color for text
        # red_color = (0, 0, 255)  # (B, G, R)   
    
        # # print(down)
        # cv2.line(frame,(282,308),(1004,308),red_color,3)  #  starting cordinates and end of line cordinates
        # cv2.putText(frame,('red line'),(280,308),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA) 


        # downwards = (len(counter_down))
        # cv2.putText(frame,('going down - ')+ str(downwards),(60,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1, cv2.LINE_AA) 
    


        # cv2.line(frame,(282,308),(1004,308),red_color,3)  #  starting cordinates and end of line cordinates
        # cv2.putText(frame,('red line'),(280,308),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)    
    
        # cv2.imshow("frames", frame)
        # cv2.imwrite(f"frames_{flag}.png", frame)
        # images.append(imageio.imread(f"frames_{flag}.png"))
        # remove(f"frames_{flag}.png")
    
        if cv2.waitKey(1)&0xFF==27:
            break
        
    count_label = set(object_counter.keys())
    count_label1 = set(object_counter1.keys())
    count_label = count_label.union(count_label1)
    count_value = {}
    for i in count_label:
        if i in object_counter:
            if i in count_value:
                count_value[i] += object_counter[i]
            else:
                count_value[i] = object_counter[i]
                        
        if i in object_counter1:
            if i in count_value:
                count_value[i] += object_counter1[i]
            else:
                count_value[i] = object_counter1[i]
                    
    plt.bar(list(count_label), list(count_value.values()), color = "maroon")
    plt.xlabel("Class name")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"outputs/count_value_{flag}.png")
    plt.close()
    # imageio.mimsave(f"outputs/output_video.gif", images)
    images.clear()
    cap.release()
    output_video.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()