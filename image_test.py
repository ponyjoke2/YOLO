import numpy as np
import cv2  # opencv包
from ultralytics import YOLO

def arrayToOpencv(image_array):
    np_array=np.array(image_array[..., ::-1])
    mat=cv2.cvtColor(np_array,cv2.COLOR_RGB2BGR)  # RGB转BGR？？[..., ::-1]
    return mat

model1 = YOLO("yolov8/best_hat_20230920.pt")
model2 = YOLO("yolov8/best_vest_230919.pt")

image_path="C:\\Users\\535756282\\Desktop\\images\\18.jpg"

res1=model1(image_path)
dst_image_array1 = res1[0].plot()
dst_image1=arrayToOpencv(dst_image_array1)
cv2.imshow("dst_image1", dst_image1)

res2=model2(image_path)
dst_image_array2 = res2[0].plot()
dst_image2=arrayToOpencv(dst_image_array2)
cv2.imshow("dst_image2", dst_image2)

cv2.waitKey()