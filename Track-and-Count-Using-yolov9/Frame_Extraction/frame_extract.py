import cv2
import numpy as np
import os
import random
from os.path import exists
from os import makedirs, remove

def main():
    p = os.listdir()
    frames = []
    for file_name in p:
        cap = cv2.VideoCapture(file_name)
        print(file_name)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frames.append(frame)
            
        cap.release()
    
    save_path = "extracted_frame"
    if not exists(save_path):
            makedirs(save_path)
    
    for i in range(100):
        output_frame = random.randint(0, len(frames))
        cv2.imwrite(f"{save_path}/{i}.png", frames[output_frame])

if __name__ == "__main__":
    main()