import math
from flask import Flask, request
import numpy as np
import cv2 #opencv包
from ultralytics import YOLO
import base64
from PIL import Image
from io import BytesIO
import urllib.request
import time

app = Flask(__name__)

model1 = YOLO("yolov8/best_hat_20230920.pt")
model2 = YOLO("yolov8/best_vest_230919.pt")
hat_ratio=0.1
hat_resize_x=160
hat_resize_y=160
vest_ratio=0.1
vest_resize_x=160
vest_resize_y=200

def saturate(val,minval,maxval):
    if val<=maxval and val>=minval:
        return val
    elif val>maxval:
        return maxval
    else:
        return minval

def readImageFromUrl(url):
    res = urllib.request.urlopen(url)
    img = np.asarray(bytearray(res.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

def opencvToBase64(image):
    ret, buffer = cv2.imencode('.jpg', image)
    image_base64string = base64.b64encode(buffer).decode('utf-8')
    return image_base64string

def base64ToOpencv(string):
    image_b64decode = base64.b64decode(string)  # base64解码
    image_array = np.fromstring(image_b64decode, np.uint8)  # 转换np序列
    image = cv2.imdecode(image_array, cv2.COLOR_BGR2RGB)  # 转换Opencv格式
    return image

def arrayToBase64(image_array):
    pil_img = Image.fromarray(image_array[..., ::-1])  # RGB转BGR？？[..., ::-1]
    buff = BytesIO()
    pil_img.save(buff, format="JPEG")
    image_base64string = base64.b64encode(buff.getvalue()).decode("utf-8")  # base64编码
    return image_base64string

def readImageFromUrl(url):
    res = urllib.request.urlopen(url)
    img = np.asarray(bytearray(res.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello World!'



#算法一，安全帽(接收url,返回检测框截图)
@app.route('/detect_post5',methods=['POST']) #POST请求接受参数
def detect_post5():  # put application's code here
    start_time=time.time()
    message = {"boxes": "", "message": "", "alert": "false"}
    try:
        imageUrl = request.get_json()["imageUrl"]
        img = readImageFromUrl(imageUrl)
        results = model1(img)  # YOLO
        print(results[0].names)
    except Exception as e:
        print("一阶段异常:", str(e))
        message["message"] = str(e)
        return message

    print("===========================================================")

    try:
        # image_array = results[0].plot()
        # dst_image_base64string = arrayToBase64(image_array)
        # dst_image = base64ToOpencv(dst_image_base64string)
        # cv2.imshow("dst_image",dst_image)

        array=results[0].boxes.data.cpu().numpy().tolist()
        print(results[0].boxes.data)

        num=len(array)
        list=[]
        for i in range(num):
            if array[i][5]==1:
                list.append(i)

        temp_message=[]
        for i in list:
            y1=math.floor(array[i][1])
            y2=math.floor(array[i][3])
            x1=math.floor(array[i][0])
            x2=math.floor(array[i][2])
            width_edge = math.floor((x2 - x1) * hat_ratio)
            height_edge = math.floor((y2 - y1) * hat_ratio)
            fy1=saturate(y1-height_edge,0,img.shape[0]-1)
            fy2 = saturate(y2 + height_edge, 0, img.shape[0]-1)
            fx1 = saturate(x1 - width_edge, 0, img.shape[1]-1)
            fx2 = saturate(x2 + width_edge, 0, img.shape[1]-1)
            #print(str(i)+"  y1:"+str(y1)+",y2:"+str(y2)+",x1:"+str(x1)+",x2:"+str(x2)+",fy1:"+str(fy1)+",fy2:"+str(fy2)+",fx1:"+str(fx1)+",fx2:"+str(fx2)+",width_edge:"+str(width_edge)+",height_edge:"+str(height_edge)+",shape0:"+str(img.shape[0])+",shape1:"+str(img.shape[1]))

            crop_img = img[fy1:fy2, fx1:fx2]
            resized_img = cv2.resize(crop_img, (hat_resize_x, hat_resize_y), interpolation=cv2.INTER_LINEAR)
            image_base64string = opencvToBase64(resized_img)

            # image = base64ToOpencv(image_base64string)
            # cv2.imshow("image"+str(i),image)

            temp_message.append(image_base64string)

        message["boxes"] = temp_message
        if len(list)>0:
            message["alert"]="true"
        #message["dst_image"]=dst_image_base64string
        message["message"]="success"

    except Exception as e:
        print("二阶段异常:", str(e))
        message["message"] = str(e)
        return message
    end_time = time.time()

    print("总运行时间：", str(end_time - start_time))
    cv2.waitKey()
    return message  # 响应json

#算法二,工作背心(接收url,返回检测框截图)
@app.route('/detect_post6',methods=['POST']) #POST请求接受参数
def detect_post6():  # put application's code here
    start_time=time.time()
    message = {"boxes": "", "message": "", "alert": "false"}

    try:
        imageUrl=request.get_json()["imageUrl"]
        img=readImageFromUrl(imageUrl)
        results = model2(img) #YOLO
        print(results[0].names)
    except Exception as e:
        print("一阶段异常:", str(e))
        message["message"]=str(e)
        return message

    print("===========================================================")

    try:
        # image_array = results[0].plot()
        # dst_image_base64string = arrayToBase64(image_array)
        # dst_image = base64ToOpencv(dst_image_base64string)
        # cv2.imshow("dst_image",dst_image)

        array=results[0].boxes.data.cpu().numpy().tolist()
        print(results[0].boxes.data)

        num=len(array)
        list=[]
        for i in range(num):
            if array[i][5]==0:
                list.append(i)

        temp_message=[]

        for i in list:
            y1=math.floor(array[i][1])
            y2=math.floor(array[i][3])
            x1=math.floor(array[i][0])
            x2=math.floor(array[i][2])
            width_edge = math.floor((x2 - x1) * vest_ratio)
            height_edge = math.floor((y2 - y1) * vest_ratio)
            fy1=saturate(y1-height_edge,0,img.shape[0]-1)
            fy2 = saturate(y2 + height_edge, 0, img.shape[0]-1)
            fx1 = saturate(x1 - width_edge, 0, img.shape[1]-1)
            fx2 = saturate(x2 + width_edge, 0, img.shape[1]-1)
            #print(str(i)+"  y1:"+str(y1)+",y2:"+str(y2)+",x1:"+str(x1)+",x2:"+str(x2)+",fy1:"+str(fy1)+",fy2:"+str(fy2)+",fx1:"+str(fx1)+",fx2:"+str(fx2)+",width_edge:"+str(width_edge)+",height_edge:"+str(height_edge)+",shape0:"+str(img.shape[0])+",shape1:"+str(img.shape[1]))

            crop_img = img[fy1:fy2, fx1:fx2]
            resized_img = cv2.resize(crop_img, (vest_resize_x, vest_resize_y), interpolation=cv2.INTER_LINEAR)
            image_base64string = opencvToBase64(resized_img)

            # image = base64ToOpencv(image_base64string)
            # cv2.imshow("image"+str(i),image)

            temp_message.append(image_base64string)

        message["boxes"] = temp_message
        if len(list)>0:
            message["alert"]="true"
        #message["dst_image"]=dst_image_base64string
        message["message"]="success"

    except Exception as e:
        print("二阶段异常:", str(e))
        message["message"] = str(e)
        return message
    end_time = time.time()
    print("总运行时间：",str(end_time - start_time))
    cv2.waitKey()
    return message  # 响应json
if __name__ == '__main__':
    app.run()
