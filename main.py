import time
import cv2 as cv
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from pathlib import Path

# limit the number of cpus used by high performance libraries
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import shutil

import sys
sys.path.insert(0, './object_detection')
from object_detection.detectors.yolov5.detector import Detector

# sys.path.insert(0, './object_detection/yolov5')
from object_detection.yolov5.utils.datasets import LoadImages, LoadStreams
from object_detection.yolov5.utils.general import check_imshow
from object_detection.yolov5.utils.torch_utils import time_sync
from object_detection.yolov5.utils.plots import Annotator, colors 
from object_detection.yolov5.utils.general import LOGGER, check_img_size, non_max_suppression, scale_coords, check_imshow, xyxy2xywh, increment_path
                           

PATH_WEIGHT_YOLOV5 = 'yolov5m6.pt'
PATH_SOURCE ='test01.mp4'

# PATH_SAVE_MP4="outputs/t1.mp4"
PATH_SAVE_FOLDER = 'outputs'
SAVE_NAME = 'y1-02'

def main():
    print('cuda is available ?', torch.cuda.is_available())
    # paremeter to run file main
    show_vid = True
    webcam = False
    source = PATH_SOURCE
    # save_path = PATH_SAVE_MP4
    save_vid = True
    visualize = False
    out = PATH_SAVE_FOLDER
    # evaluate = False
  
    # # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
    # # its own .txt file. Hence, in that case, the output folder is not restored
    # if not evaluate:
    #     if os.path.exists(out):
    #         pass
    #         shutil.rmtree(out)  # delete output folder
    #     os.makedirs(out)  # make new output folder
    
    # create Detector is Yolov5 classes=(2, 5, 6, 7, 8)

    detector = Detector(ckpt=PATH_WEIGHT_YOLOV5, conf_thres=0.5, classes=(2, 5, 6, 7, 8))

    
    # Set Dataloader
    vid_path, vid_writer = None, None
    # Check if environment supports image displays
    if show_vid:
        show_vid = check_imshow()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        source = '0'
        dataset = LoadStreams(source, img_size=detector.imgsz, stride=detector.stride, auto=detector.pt and not detector.jit)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=detector.imgsz, stride=detector.stride, auto=detector.pt and not detector.jit)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    save_path = str(Path(out))
    # extract what is in between the last '/' and last '.'
    txt_file_name = source.split('/')[-1].split('.')[0]
    txt_path = str(Path(out)) + '/' + txt_file_name + '.txt'
    
    dt, seen = [0.0, 0.0, 0.0], 0
    
    
    
    t0 = time.time()
    # Start Process
    frame_count = 0
    for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):
        img_per = im0s.copy()
        time_start = time.time()
        t1 = time_sync()
        img = torch.from_numpy(img).to(detector.device)
        img = img.half() if detector.half else img.float()
        img /= 255.0 # 0 - 255 to 0.0 - 1.0
        
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
            
        t2 = time_sync()
        dt[0] += t2 - t1
        # Inference
        visualize = increment_path(save_path / Path(path).stem, mkdir=True) if visualize else False
        t3 = time_sync()
        dt[1] += t3 - t2
        
        pred, dt[1], dt[2], t3 = detector.detect(img=img, dt=dt, t2=t2, visualize=visualize)

        
        for i, det in enumerate(pred):
            seen += 1
            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            s += '%gx%g' % img.shape[2:] # ex. print string
            name_file = SAVE_NAME + ".mp4"
            save_path = str(Path(out) / name_file) 
            
            annotator = Annotator(im0, line_width=2, pil=not ascii)
            
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                
                if((frame_idx % 20) == 0 ):
                    for idx, (*xyxy, conf, cls) in enumerate(reversed(det)):
                        x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]) 

                        print('car{}'.format(frame_count))

                        crop_img = img_per[y1:y2, x1:x2]
                
                        cv.imwrite('cars/all/{}-{}.jpg'.format(frame_count, idx), crop_img)
                    
                
                
                # Write results
                if show_vid :
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        # label = None if detector.hide_labels else (detector.names[c] if detector.hide_conf else f'{detector.names[c]} "sdsd" {conf.cpu(): .2f}')
                        label = f'{detector.names[c]} {conf: .2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
    
            else:
                print("None Object detection")
            
            # Stream results
            im0 = annotator.result()
            
            # show data in image
            fps = 1./(time.time()-time_start)
            cv.putText(im0, "FPS: {:.2f}".format(fps), (5,30), cv.FONT_HERSHEY_SIMPLEX, 0.75, (255,0,0), 2)
         
            
        if show_vid:
            cv.imshow(p, im0)
            if cv.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration
            

        # Save results (image with detections)
        if save_vid:
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv.VideoWriter):
                    vid_writer.release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path += '.mp4'

                vid_writer = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer.write(im0)
        frame_count += 1
    
    print('Done. (%.3fs)' % (time.time() - t0))

                
    
    
if __name__ == "__main__":
    main()