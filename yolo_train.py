from ultralytics import YOLO
import os
if __name__=='__main__':
    model = YOLO("yolo11m-cls.pt")
    model.train(data="datasets",epochs=50, imgsz=160, device=0)  # train
    model.save("myyolo11m-cls8.pt")  # save