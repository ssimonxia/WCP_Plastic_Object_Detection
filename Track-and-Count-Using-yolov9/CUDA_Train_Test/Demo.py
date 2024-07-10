from roboflow import Roboflow
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from ultralytics import YOLO

def main():
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    model = YOLO('yolov8m.pt')
    results = model.train(data="./datasets/data.yaml", epochs = 8, workers = 2, batch = 16, imgsz = 320)
    print(results)
    
if __name__ == "__main__":
    main()
