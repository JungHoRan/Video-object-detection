# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run inference on images, videos, directories, streams, etc.

Usage:
    $ python path/to/detect.py --weights yolov5s.pt --source 0  # webcam
                                                             img.jpg  # image
                                                             vid.mp4  # video
                                                             path/  # directory
                                                             path/*.jpg  # glob
                                                             'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                             'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
"""
##导入安装好的python库
import argparse
import os
import sys
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn

##获取当前的绝对路径
FILE = Path(__file__).resolve()  #__file__指的是当前文件(这里就是detect.py)"E:\yolov5_obb-master\yolov5_obb-master\detect.py"
ROOT = FILE.parents[0]  # YOLOv5 root directory，获取父目录，"E:\yolov5_obb-master"
if str(ROOT) not in sys.path:  #sys.path即当前python环境可以运行的路径，项目不在该路径就无法运行其中模块
    sys.path.append(str(ROOT))  # add ROOT to PATH,把父目录添加到运行路径上
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative 设置相对路径，以便寻找项目中的文件

##加载自定义模块（上一步完成了路径加载，这里才可以导入）
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, non_max_suppression_obb, print_args, scale_coords, scale_polys, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
from utils.rboxs_utils import poly2rbox, rbox2poly

#1载入参数
@torch.no_grad()#该标注使得所有计算得出的tensor和requires_grad都自动设置为False,不进行梯度计算，也就没办法进行反向传播
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS #进行nms是否去除不同类别之间的框
        augment=False,  # augmented inference
        visualize=True,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    #2初始化配置
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))#网络流地址
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)#摄像头，isnumeric检测字符串是否由数字组成
    if is_url and is_file:
        source = check_file(source)  # download
    #3保存结果
    # Directories保存结果
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # 4Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)#选择后端框架
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Half
    #开启半精度，加快运行速度
    half &= (pt or jit or engine) and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()
    #5加载数据
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    #6、推理部分
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup，热身，先跑一遍预测流程，可以加速预测
    dt, seen = [0.0, 0.0, 0.0], 0
    #dt:存储每一步的耗时
    #seen:计数功能，已经处理完了多少帧图片
    for path, im, im0s, vid_cap, s in dataset:
        '''
        dataset中，每次迭代的返回值是self.sources,img,img0,None,''
        img是resize后的图片
        img0是原始图片
        s是图片的基本信息
        '''
        ##6.1预处理
        t1 = time_sync()#获取当前时间
        im = torch.from_numpy(im).to(device)#将图片放到指定设备上识别，将图片转换成tensor格式
        im = im.half() if half else im.float()  # uint8 to fp16/32 #将输入转化为半精度/全精度浮点数
        im /= 255  # 0 - 255 to 0.0 - 1.0 #除255进行归一化
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim #增加一个维度
        t2 = time_sync()
        dt[0] += t2 - t1 #记录该阶段的耗时

        ##6.2对每张图片/视频进行前向推理
        # Inference
        #可视化的话，就保留推理过程中的特征图
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        #推理结果，pred保存的是所有bound_box的信息
        pred = model(im, augment=augment, visualize=visualize)#augment指示预测时是否应该使用数据增强
        t3 = time_sync()
        dt[1] += t3 - t2

        ##6.3NMS除去多余的框
        # NMS
        # pred: list*(n, [xylsθ, conf, cls]) θ ∈ [-pi/2, pi/2)
        #返回值为过滤后的预测框
        pred = non_max_suppression_obb(pred, conf_thres, iou_thres, classes, agnostic_nms, multi_label=True, max_det=max_det)
        '''
        pred:上一步网络的输出结果
        conf_thres:置信度阈值
        iou_thres:iou阈值
        classes:是否只保留特定的类别
        agnostic_nms：是否去除不同类别之间的框，不同类别的框是否同时处理
        '''
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        ##6.4预测过程
        # Process predictions
        #把所有检测框画到原图中
        for i, det in enumerate(pred):  # per image
            '''
            i是batch信息
            det是检测框的信息
            '''
            pred_poly = rbox2poly(det[:, :5]) # (n, [x1 y1 x2 y2 x3 y3 x4 y4])#取前5列，将矩形框的坐标角度转换成多边形的顶点坐标
            seen += 1

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string #添加图像的尺寸信息
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh #归一化增益
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))#创建了一个Annotator对象，便于在图像上绘制检测结果

            if len(det):
                # Rescale polys from img_size to im0 size#将预测图信息映射回原图
                # det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                pred_poly = scale_polys(im.shape[2:], pred_poly, im0.shape)
                det = torch.cat((pred_poly, det[:, -2:]), dim=1) # (n, [poly conf cls])

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

            ##6.5打印目标检测结果
                # Write results
                for *poly, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        #未旋转的det信息中包含xyxy(左上角+右下角)格式转换为xywh(中心点+宽长)格式，并归一化
                        # xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        poly = poly.tolist()
                        #print("poly是什么：",poly)poly是四个点的坐标
                        #line的形式是“类别 x y w h”,若save_conf为true,则line形式是“类别 x y w h 置信度”
                        line = (cls, *poly, conf) if save_conf else (cls, *poly)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    #在原图上画+将预测到的目标剪切出来保存成图片
                    if save_img or save_crop or view_img:  # Add poly to image
                        c = int(cls)  # integer class #类别标号
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')#类别
                        # annotator.box_label(xyxy, label, color=colors(c, True))
                        annotator.poly_label(poly, label, color=colors(c, True))#绘制旋转框
                        if save_crop: # Yolov5-obb doesn't support it yet
                            # save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            pass

            # Print time (inference-only)

            ##6.6在窗口中实时查看检测结果
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond
            #6.7设置保存结果
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / "E:\yolov5_obb-master\yolov5_obb-master\data\\best.pt", help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'E:\yolov5_obb-master\yolov5_obb-master\dataset\P0909.jpg', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[840], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')#最大检测框数量
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')#是否覆盖已有结果
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt) #打印参数信息
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))#检测包是否成功安装，requirement.txt的包是否安装
    run(**vars(opt))#将opt变量的属性和属性值作为关键字参数传递给run()函数


if __name__ == "__main__":
    opt = parse_opt() #解析参数
    main(opt)
