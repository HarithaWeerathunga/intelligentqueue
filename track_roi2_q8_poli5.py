import sys
sys.path.insert(0, './yolov5')

from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn


from customfunctions import is_inside_polygon, peopleCounter, peopleCounterQueue , peopleCounterReturn, peopleCounterQueueReturn, calcAvgWaitingTime
import datetime
import numpy as np
from db import insertCashierData, insertQueueData

#cashier q8

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

#queue ROI
rx1 = 800
ry1 = 0
rx2 = 1100
ry2 = 1000

#cashier ROI #1
c1x1 = 300
c1y1 = 500
c1x2 = 650
c1y2 = 900


#cashier ROI #2
cx1 = 1150
cy1 = 500
cx2 = 1450
cy2 = 870

cashierROI = [(1150,500),(1450,500), (1450,750), (1300,850),(1150,870)]
queueROI = [(900,0),(1300,0), (1300,400), (1100,600),(1100,1200),(600,1200),(650,320)]





def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, identities=None, offset=(0, 0)):


    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color_bbox = compute_color_for_labels(id)
        label = 'PID : {}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

        # Region of interest queue
        ROI = [(rx1,ry1),(rx2,ry2)]
        

        #cashier 2 polygon
        c1_pts = np.array([[1150, 500], [1450, 500], 
                [1450,750], [1300,850], [1150,870]],
               np.int32)
        c1_pts = c1_pts.reshape((-1,1,2))
        isClosed = True 
        color = (0,0,255)
        thickness = 3
        cv2.polylines(img,[c1_pts],isClosed,color,thickness)
        cashierROI = [(1150,500),(1450,500), (1450,750), (1300,850),(1150,870)]
        bx = (x1+x2)/2
        by = (y1+y2)/2
        bboxCoord = (bx,by)
        cashierPoli = is_inside_polygon(cashierROI, bboxCoord)



        if cashierPoli:
            cv2.putText(img, "Cashier Present", (1210,480), cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, color= (0,255,0) , thickness=3)
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,255), 2)
            cv2.putText(img, "Cashier 2", (x1, y1 +
                                    t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
            # insertCashierData("2", datetime.datetime.now(), "Available")
        else:
            print("ok")
            # cv2.putText(img, "Cashier Absent", (1210,480), cv2.FONT_HERSHEY_SIMPLEX, fontScale= 1, color= (0,255,0) , thickness=3)
            # insertCashierData("2", datetime.datetime.now(), "Not Available")

        #queue polygon
        queueROI = [(900,0),(1300,0), (1300,400), (1100,600),(1100,1200),(600,1200),(650,320)]
        queue_pts = np.array([[900, 0], [1300, 0], 
                [1300,400], [1100,600], [1100,1200],[600,1200],[650,320]],
               np.int32)
        queue_pts = queue_pts.reshape((-1,1,2))
        isClosed = True 
        
        thickness = 3
        cv2.polylines(img,[queue_pts],isClosed,color,thickness)

        queuePoli = is_inside_polygon(queueROI, bboxCoord)

        

        if queuePoli:
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color_bbox, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color_bbox, -1)
            cv2.putText(img, label, (x1, y1 +
                                    t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        
        
    return img



def ray_tracing(x1,y1,x2,y2,rx1,ry1,rx2,ry2):
    x = (x1+x2) / 2 
    y = (y1+y2) / 2
    if (x > rx1 and x < rx2) and (y > ry1 and y < ry2):
        return 1
    else :
        return 0 



def detect(opt, save_img=False):
    out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')

    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)
    deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                        max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    # run once
    _ = model(img.half() if half else img) if device.type != 'cpu' else None

    save_path = str(Path(out))
    txt_path = str(Path(out)) + '/results.txt'


    # all the objects ids we have tracked or detected
    object_id_list = []
    # holds current date time for a particular person
    dtime = dict()
    dwell_time = dict()

   
    
   

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

        
        

        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)


        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        # x_np = torch.from_numpy(im0s)
        # pred = model(x_np, augment=opt.augment)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()


        total_people_in_frame = 0
        

        
        # #person count
        # for i,det in enumerate(pred):
            
        #     peopleCounter(det, im0s)

        #people counter return
        for i,det in enumerate(pred):
            
            total_people_in_frame = peopleCounterReturn(det, im0s)
            
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image

            

            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            s += '%gx%g ' % img.shape[2:]  # print string
            save_path = str(Path(out) / Path(p).name)

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(
                    img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                bbox_xywh = []
                confs = []

                # Adapt detections to deep sort input format
                for *xyxy, conf, cls in det:
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf.item()])


                indexes_to_remove = []
                indexes_to_keep = []

                for i in range(len(bbox_xywh)):
                    print("printing bbox array")
                    print(bbox_xywh)
                    print(bbox_xywh[i])
                    x1 = bbox_xywh[i][0]
                    y1 = bbox_xywh[i][1]
                    x2 = bbox_xywh[i][2]
                    y2 = bbox_xywh[i][3]
                    rt = ray_tracing(x1,y1,x2,y2,rx1,ry1,rx2,ry2)

                    bx = (x1+x2)/2
                    by = (y1+y2)/2
                    bboxCoordCount = (bx,by)

                    queuePoli = is_inside_polygon(queueROI, bboxCoordCount)



                    if queuePoli:
                        print("inside")
                        indexes_to_keep.append(i)
                        
                    else :
                        print("outside")
                        indexes_to_remove.append(i)
                
                peopleCounterQueue(indexes_to_keep, im0s)

                # person count
                for i,det in enumerate(indexes_to_keep):
                    
                    peopleCounterQueueReturn(indexes_to_keep, im0s, total_people_in_frame,"1")

  


                roi_list = []

                for element in indexes_to_keep:
                    roi_list.append(bbox_xywh[element])
                
                print("needs to keep")
                print(len(roi_list))
                print(roi_list)
                print(bbox_xywh)
                
                
                


                xywhs = torch.Tensor(bbox_xywh)
                # xywhs = torch.Tensor(roi_list)
                confss = torch.Tensor(confs)
                
               

                # Pass detections to deepsort
                outputs = deepsort.update(xywhs, confss, im0)

                print("Print outputs")
                print(outputs)

                #print confidence levels

                waiting_time = 0
                

                for i in outputs:

                    x1 = i[0]
                    y1 = i[1]
                    x2 = i[2]
                    y2 = i[3]
                    track_id = i[4]
                    
                    
                    bx = (x1+x2)/2
                    by = (y1+y2)/2
                    bboxCoord = (bx,by)

                    queuePoli = is_inside_polygon(queueROI, bboxCoord)

                    


                    if queuePoli:
                            
                            if track_id not in object_id_list:
                                object_id_list.append(track_id)
                                dtime[track_id] = datetime.datetime.now()
                                dwell_time[track_id] = 0
                                print("If")
                                # insertQueueData(track_id.item(), datetime.datetime.now())
                                
                                
                            else: 
                                curr_time = datetime.datetime.now()
                                old_time = dtime[track_id]
                                diff_time = curr_time -  old_time
                                dtime[track_id] = datetime.datetime.now()
                                sec = diff_time.total_seconds()
                                #sec = sec / 10
                                dwell_time[track_id] += sec
                                print("waiting time***************************************************************************")
                                waiting_time += sec
                                
                                
                                print("Else")
                                # insertQueueData(track_id.item(), datetime.datetime.now())
                            
                            dwell_time_text = "Wait Time : {}".format(int(dwell_time[track_id]))
                            person_id_text = "Person ID : {}".format(track_id)
                            # cv2.putText(im0s,  dwell_time_text ,(int(bbox_top), int(bbox_top-10)),0, 0.75, (255,255,255),2)
                            #image,text,(x,y),fontscale,color,thickness,(x1,y1-10)
                            cv2.putText(im0s,  dwell_time_text ,(x1+50,y1+110),cv2.FONT_HERSHEY_SIMPLEX, 0.80, (16,16,227),3)
                            #cv2.putText(im0s,  "Person ID" ,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0),2)
                        
                
                    



                # draw boxes for visualization
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    print("bbox_xyxy")
                    print(bbox_xyxy)
                    identities = outputs[:, -1]
                    draw_boxes(im0, bbox_xyxy, identities)

                # Write MOT compliant results to file
                if save_txt and len(outputs) != 0:
                    for j, output in enumerate(outputs):
                        bbox_left = output[0]
                        bbox_top = output[1]
                        bbox_w = output[2]
                        bbox_h = output[3]
                        identity = output[-1]
                        with open(txt_path, 'a') as f:
                            f.write(('%g ' * 10 + '\n') % (frame_idx, identity, bbox_left,
                                                           bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))  # label format

            else:
                deepsort.increment_ages()
            
            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
    

        




    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/23_best.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=640,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    # class 0 is person
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[0], help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    parser.add_argument("--config_deepsort", type=str,
                        default="deep_sort_pytorch/configs/deep_sort.yaml")
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)

    with torch.no_grad():
        detect(args)
