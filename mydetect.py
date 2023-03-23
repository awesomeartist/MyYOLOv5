import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh,xywh2xyxy, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def detect(save_img=False):
    #接收输入参数
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    #生成存储文件夹
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    #获取检测设备
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    #加载Float32模型，确保用户设定的输入图片分辨率能整除32(如不能整除则调整为能整除并返回)
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    #设置第二次分类，默认不使用
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    #通过不同的输入源来设置不同的数据加载方式
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    # 获取类名
    names = model.module.names if hasattr(model, 'module') else model.names
    #设置画框颜色
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    """
    path 图片/视频路径
    img 进行resize+pad之后的图片
    img0 原size图片
    cap 当读取图片时为None，读取视频时为视频源
    """
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        """
        前向传播 返回pred的shape是(1, num_boxes, 5+num_class)
        h,w为传入网络图片的长和宽，注意dataset在检测时使用了矩形推理，所以这里h不一定等于
        num_boxes = h/32 * w/32 + h/16 * w/16 + h/8 * w/8
        pred[..., 0:4]为预测框坐标
        预测框坐标为xywh(中心点+宽长)格式
        pred[..., 4]为objectness置信度
        pred[..., 5:-1]为分类结果
        """
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # Apply NMS
        """
        pred:前向传播的输出
        conf_thres:置信度阈值
        iou_thres:iou阈值
        classes:是否只保留特定的类别
        agnostic:进行nms是否也去除不同类别之间的框
        经过nms之后，预测框格式：xywh-->xyxy(左上角右下角)
        pred是一个列表list[torch.tensor]，长度为batch_size
        每一个torch.tensor的shape为(num_boxes, 6),内容为box+conf+cls
        """
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # Apply Classifier
        #添加二次分类默认不使用
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        #检测
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            #检测文件路径
            p = Path(p)  # to Path
            #存储路径
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            #打印图片大小
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                #打印检测结果中的类别数量与类名
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f'{n} {names[int(c)]}s, '  # add to string

                # Write results
                #记录检测结果
                classes = []
                xy_xy = []
                xy_wh = []
                helmet = []
                safetybelt = []
                worker = []
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    xyxy = (xywh2xyxy(torch.tensor(xywh).view(1, 4)) ).view(-1).tolist()
                    xy_xy.extend(xyxy)
                    xy_wh.extend(xywh)
                    classes.append(names[int(cls)])
                for i, obj0 in enumerate(classes):
                    if obj0 == 'helmet':
                        helmet.extend(xy_wh[i])
                    if obj0 == 'safetybelt':
                        safetybelt.aextend(xy_wh[i])
                    else:
                        worker.aextend(xy_xy[i])   

                def check_helmet():
                    flag = True
                    break1 = False
                    for x in worker:
                        # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
                        x1 = x[0]
                        y1 = x[1]
                        x2 = x[2]
                        y2 = x[3]
                        if helmet:#判断列表是否为空
                            for x0 in helmet:
                                # y = x0.clone() if isinstance(x0, torch.Tensor) else np.copy(x0)
                                x3 = x0[0]
                                y3 = x0[1]
                                if x3 > x1 and x3 < x2 and y3 > y1 and y3 < y2:
                                    break
                                elif x0 == helmet[-1]:
                                    break1 = True
                        else:
                            break1 = False
                        if break1:
                            flag  = False
                            break                           
                    return flag


                def check_safetybelt():
                    flag = True
                    if 'on hight1' in classes:
                        x = xy_xy[classes.index('on hight1')]
                    else:
                        x = xy_xy[classes.index('on high2')]
                    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
                    x1 = x[0]
                    y1 = x[1]
                    x2 = x[2]
                    y2 = x[3]
                    if safetybelt:  #判断列表是否为空
                        for x0 in safetybelt:
                            # y = x0.clone() if isinstance(x0, torch.Tensor) else np.copy(x0)
                            x3 = x0[0]
                            y3 = x0[1]
                            if x3 > x1 and x3 < x2 and y3 > y1 and y3 < y2:
                                break
                            elif x0 == helmet[-1]:
                                break1 = True
                    else:
                        break1 = False
                    if break1:
                        flag = False
                    return flag


                def check_ladder():

                    return 'hold ladder' in classes

                 
                for *xyxy, conf, cls in reversed(det):
                    #接收输入存储文本指令后触发
                    if save_txt:  # Write to file
                        # 将xyxy(左上角+右下角)格式转为xywh(中心点+宽长)格式，并除上w，h做归一化，转化为列表再保存
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #接收到存储图片或显示命令后在原图上进行画框
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                #图像上显示检测结果
                if 'on hight1' in classes:
                    cv2.putText(im0, 'work outside and on high', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 200, 0), 3)
                    if check_helmet():                  
                        if check_safetybelt() and check_ladder():
                            cv2.putText(im0, 'safe', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 200, 0), 3)
                        elif check_safetybelt() and not check_ladder():
                            cv2.putText(im0, 'work on high alone', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                        elif not check_safetybelt() and check_ladder():
                            cv2.putText(im0, 'no safetybelt', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                        else:
                            cv2.putText(im0, 'no safetybelt', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                            cv2.putText(im0, 'work on high alone', (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                    else:
                        cv2.putText(im0, 'no helmet', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                        if check_safetybelt() and not check_ladder():
                            cv2.putText(im0, 'work on high alone', (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                        elif not check_safetybelt and check_ladder():
                            cv2.putText(im0, 'no safetybelt', (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                        elif not check_safetybelt and not check_ladder():
                            cv2.putText(im0, 'no safetybelt', (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                            cv2.putText(im0, 'work on high alone', (50, 250), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)

                if 'beside cabinet' in classes:
                    cv2.putText(im0, 'work beside cabinet', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 200, 0), 3)
                    if check_helmet():
                        cv2.putText(im0, 'safe', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 200, 0), 3)
                    else:
                        cv2.putText(im0, 'no helmet', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)


                if 'on high2' in classes:
                    cv2.putText(im0, 'work inside and on high', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 200, 0), 3)
                    if  check_safetybelt() and check_helmet():
                        cv2.putText(im0, 'safe', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 200, 0), 3)
                    elif check_safetybelt() and not check_helmet():
                        cv2.putText(im0, 'no helmet', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                    elif not check_safetybelt() and check_helmet():
                        cv2.putText(im0, 'no safetybelt', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                    else:
                        cv2.putText(im0, 'no helmet', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)
                        cv2.putText(im0, 'no safetybelt', (50, 200), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)


                if 'work outside' in classes:
                    cv2.putText(im0, 'work outside', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 200, 0), 3)
                    if 'on car' in classes:
                        cv2.putText(im0, 'work on car', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 200, 0), 3)
                    if check_helmet():
                        cv2.putText(im0, 'safe', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 200, 0), 3)
                    else:
                        cv2.putText(im0, 'no helmet', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 200), 3)


                elif 'worker' in classes:
                    if check_helmet():
                        cv2.putText(im0, 'safe!', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 4.0, (0, 200, 0), 3)
                    else:
                        cv2.putText(im0, 'no helmet', (50, 150), cv2.FONT_HERSHEY_COMPLEX, 4.0, (0, 0, 200), 3)
                        
                        
            # Print time (inference + NMS)
            #打印处理该帧图片的时间
            print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            #如果设置展示，则show图片/视频
            if view_img:
                cv2.imshow(str(p), im0)

            # Save results (image with detections)
            #设置保存图片/视频
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video'
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
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        # 打印结果存储路径
        print(f"Results saved to {save_dir}{s}")
    #打印总用时
    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 加载训练好的权重路径
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/myworks3_1/weights/best.pt', help='model.pt path(s)')
    # 设置检测数据，文件路径（单张图片或者存放检测图片的文件夹）/0为本机摄像头/视频流
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    # 输入图片的大小
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    #置信度阈值
    parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
    #做NMS的IoU阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    #检测设备
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    #是否展示预测之后的图片/视频，默认False
    parser.add_argument('--view-img', action='store_true', help='display results')
    #存储检测结果的指令
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    #存储检测到类的自信度
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    #设置检测的类
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    #进行NMS是否也去除不同类别之间的框，默认False
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    #推理的时候进行多尺度、翻转等操作(TTA)推理
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    #如果为True，则对所有模型进行strip_optimizer操作，去除pt文件中的优化器等信息，默认为False
    parser.add_argument('--update', action='store_true', help='update all models')
    #设置检测结果的项目存储路径
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    #在存储路径下生成本次检测的存储文件夹名
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
