import sys
import argparse
import os

sys.path.append(r'E:\GitHubRepo\PR\ultralyticsPro-mango') # Path

from ultralytics import YOLO

def main(opt):
    yaml = opt.cfg
    model = YOLO(yaml).load('yolov8n-pose.pt') 

    model.info()
    
    # 姿态估计训练
    results = model.train(
        data='coco8-pose.yaml', 
        epochs=100, 
        imgsz=640, 
        workers=0, 
        batch=2
        )

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default= r'ultralytics\cfg\models\v8\yolov8-pose.yaml', help='initial weights path')
    parser.add_argument('--weights', type=str, default='', help='')

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)