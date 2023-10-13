import logging
import math
from flask import Flask, request
import numpy as np
import cv2  # opencv包
from ultralytics import YOLO
import base64
from PIL import Image
from io import BytesIO
import urllib.request
import time
from flasgger import Swagger


# 做了一个安全帽和安全服的目标检测算法后端项目，使用python语言开发，框架使用的是flask。算法使用的是YOLOv8。一共做了两个接口，分别对应安全帽和安全服的检测。
# 接收的参数为图片的url，返回告警图片，没带安全帽或没穿安全服的base64截图。

app = Flask(__name__)
Swagger(app)

app.config['SWAGGER'] = {
    'title': 'My API',
    'description': 'API for my data',
    'version': '1.0.0',
    'license': {
        'name': 'Apache 2.0',
        'url': 'http://www.apache.org/licenses/LICENSE-2.0.html'
    }
}

# 配置日志级别
app.logger.setLevel(logging.DEBUG)
# 日志处理程序 输出到控制台
handler = logging.StreamHandler()
#formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#handler.setFormatter(formatter)
app.logger.addHandler(handler)


model1 = YOLO("yolov8/best_hat_20230920.pt")
model2 = YOLO("yolov8/best_vest_230919.pt")
hat_ratio = 0.1
hat_resize_x = 200
hat_resize_y = 200
vest_ratio = 0.1
vest_resize_x = 200
vest_resize_y = 200


def saturate(val, minval, maxval):
    if val <= maxval and val >= minval:
        return val
    elif val > maxval:
        return maxval
    else:
        return minval




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

def arrayToOpencv(image_array):
    np_array=np.array(image_array[..., ::-1])
    mat=cv2.cvtColor(np_array,cv2.COLOR_RGB2BGR)  # RGB转BGR？？[..., ::-1]
    return mat



@app.route('/hello')
def hello_world():  # put application's code here
    """
            This is an example of hello world
            ---
            responses:
              200:
                description: A simple hello world response
                content:
                  text/plain:
                    schema:
                      type: string
    """
    return 'Hello World!'


# 算法一，安全帽(接收url,返回检测框截图)
@app.route('/detect_post5', methods=['POST'])  # POST请求接受参数
def detect_post5():  # put application's code here
    '''
    tags:
        - 安全帽
    requestBody:
        description: body
        required: true
        content:
            x-www-form-urlencoded:
            schema:
                example: {"username":TOKEN,"password":PASSWORD,"repassword":PASSWORD,"email":EMAIL,"csrf_token":TOKEN}
    responses:
        200:
            description: index test
            content:
                raw/html:
                    schema:
                        example: <!doctype html><html><head><meta charset="UTF-8" /><title>phpwind 9.0 - Powered by phpwind</title>
    '''
    start_time = time.time()
    print("********************************安全帽检测算法********************************")
    message = {"code":200,"boxes": "", "message": "", "alarmType": ""}

    # 源ip
    print("源ip:"+str(request.remote_addr))

    try:
        request_data_dict=request.get_json(silent=True)

        imageUrl = request_data_dict["imageUrl"]
        print("图片url:" + imageUrl)
        # imageUrl = "http://" + request.remote_addr + imageUrl
        img = readImageFromUrl(imageUrl)


        importantLowThreshold = int(request_data_dict["importantLowThreshold"])
        urgentLowThreshold = int(request_data_dict["urgentLowThreshold"])
        alarmThreshold = int(request_data_dict["alarmThreshold"])
        print(request_data_dict)
        if "src_x1" in request_data_dict and "src_y1" in request_data_dict and "src_x2" in request_data_dict and "src_y2" in request_data_dict:
            print("画框截图")
            src_x1 = int(request_data_dict["src_x1"])
            src_y1 = int(request_data_dict["src_y1"])
            src_x2 = int(request_data_dict["src_x2"])
            src_y2 = int(request_data_dict["src_y2"])
            img = img[src_y1:src_y2,src_x1:src_x2]

        results = model1(img)  # YOLO
        print("类别:"+str(results[0].names))
    except Exception as e:
        app.logger.debug("读取或检测图片异常:"+str(e))
        message["message"] = str(e)
        return message


    try:
        # dst_image_array = results[0].plot()
        # dst_image=arrayToOpencv(dst_image_array)
        # #dst_image=cv2.resize(dst_image,(700,700))
        # cv2.imshow("dst_image", dst_image)

        array = results[0].boxes.data.cpu().numpy().tolist()
        print("检测数据:"+str(results[0].boxes.data))

        # {0: 'hat', 1: 'person'}
        num = len(array)
        list = []
        for i in range(num):
            if array[i][5] == 1 and array[i][4] > alarmThreshold / 100:
                list.append(i)

        temp_message = []
        for i in list:
            y1 = math.floor(array[i][1])
            y2 = math.floor(array[i][3])
            x1 = math.floor(array[i][0])
            x2 = math.floor(array[i][2])
            width_edge = math.floor((x2 - x1) * hat_ratio)
            height_edge = math.floor((y2 - y1) * hat_ratio)
            fy1 = saturate(y1 - height_edge, 0, img.shape[0] - 1)
            fy2 = saturate(y2 + height_edge, 0, img.shape[0] - 1)
            fx1 = saturate(x1 - width_edge, 0, img.shape[1] - 1)
            fx2 = saturate(x2 + width_edge, 0, img.shape[1] - 1)
            # print(str(i)+"  y1:"+str(y1)+",y2:"+str(y2)+",x1:"+str(x1)+",x2:"+str(x2)+",fy1:"+str(fy1)+",fy2:"+str(fy2)+",fx1:"+str(fx1)+",fx2:"+str(fx2)+",width_edge:"+str(width_edge)+",height_edge:"+str(height_edge)+",shape0:"+str(img.shape[0])+",shape1:"+str(img.shape[1]))

            crop_img = img[fy1:fy2, fx1:fx2]
            resized_img = cv2.resize(crop_img, (hat_resize_x, hat_resize_y), interpolation=cv2.INTER_LINEAR)
            image_base64string = opencvToBase64(resized_img)

            # image = base64ToOpencv(image_base64string)
            # cv2.imshow("image" + str(i), image)

            temp_message.append(image_base64string)

        message["boxes"] = temp_message
        if len(list) == 0:
            message["alarmType"] = "无告警"
        elif 0 < len(list) <= importantLowThreshold:
            message["alarmType"] = "一般"
        elif importantLowThreshold < len(list) <= urgentLowThreshold:
            message["alarmType"] = "重要"
        elif len(list) > urgentLowThreshold:
            message["alarmType"] = "紧急"
        # message["dst_image"]=dst_image_base64string
        message["message"] = "success"

    except Exception as e:
        app.logger.debug("检测后数据处理异常:"+str(e))
        message["message"] = str(e)
        message["code"]= 500
        return message
    end_time = time.time()

    app.logger.debug("总运行时间："+str(end_time - start_time))
    # cv2.waitKey()
    return message  # 响应json


# 算法二,工作背心(接收url,返回检测框截图)
@app.route('/detect_post6', methods=['POST'])  # POST请求接受参数
def detect_post6():  # put application's code here
    start_time = time.time()
    print("********************************安全服检测算法********************************")
    message = {"code":200,"boxes": "", "message": "", "alarmType": ""}

    # 源ip
    print("源ip:"+str(request.remote_addr))

    try:
        request_data_dict = request.get_json(silent=True)

        imageUrl = request_data_dict["imageUrl"]
        print("图片url:" + imageUrl)
        # imageUrl = "http://" + request.remote_addr + imageUrl
        img = readImageFromUrl(imageUrl)

        importantLowThreshold = int(request_data_dict["importantLowThreshold"])
        urgentLowThreshold = int(request_data_dict["urgentLowThreshold"])
        alarmThreshold = int(request_data_dict["alarmThreshold"])
        print(request_data_dict)
        if "src_x1" in request_data_dict and "src_y1" in request_data_dict and "src_x2" in request_data_dict and "src_y2" in request_data_dict:
            print("画框截图")
            src_x1 = int(request_data_dict["src_x1"])
            src_y1 = int(request_data_dict["src_y1"])
            src_x2 = int(request_data_dict["src_x2"])
            src_y2 = int(request_data_dict["src_y2"])
            img = img[src_y1:src_y2, src_x1:src_x2]

        results = model2(img)  # YOLO
        print("类别:"+str(results[0].names))
    except Exception as e:
        app.logger.debug("读取或检测图片异常:"+ str(e))
        message["message"] = str(e)
        return message


    try:
        # dst_image_array = results[0].plot()
        # dst_image=arrayToOpencv(dst_image_array)
        # #dst_image=cv2.resize(dst_image,(700,700))
        # cv2.imshow("dst_image", dst_image)

        array = results[0].boxes.data.cpu().numpy().tolist()
        print("检测数据:"+str(results[0].boxes.data))

        # {0: 'others', 1: 'work vest'}
        num = len(array)
        list = []
        for i in range(num):
            if array[i][5] == 0 and array[i][4] > alarmThreshold / 100:
                list.append(i)

        temp_message = []

        for i in list:
            y1 = math.floor(array[i][1])
            y2 = math.floor(array[i][3])
            x1 = math.floor(array[i][0])
            x2 = math.floor(array[i][2])
            width_edge = math.floor((x2 - x1) * vest_ratio)
            height_edge = math.floor((y2 - y1) * vest_ratio)
            fy1 = saturate(y1 - height_edge, 0, img.shape[0] - 1)
            fy2 = saturate(y2 + height_edge, 0, img.shape[0] - 1)
            fx1 = saturate(x1 - width_edge, 0, img.shape[1] - 1)
            fx2 = saturate(x2 + width_edge, 0, img.shape[1] - 1)
            # print(str(i)+"  y1:"+str(y1)+",y2:"+str(y2)+",x1:"+str(x1)+",x2:"+str(x2)+",fy1:"+str(fy1)+",fy2:"+str(fy2)+",fx1:"+str(fx1)+",fx2:"+str(fx2)+",width_edge:"+str(width_edge)+",height_edge:"+str(height_edge)+",shape0:"+str(img.shape[0])+",shape1:"+str(img.shape[1]))

            crop_img = img[fy1:fy2, fx1:fx2]
            resized_img = cv2.resize(crop_img, (vest_resize_x, vest_resize_y), interpolation=cv2.INTER_LINEAR)
            image_base64string = opencvToBase64(resized_img)

            # image = base64ToOpencv(image_base64string)
            # cv2.imshow("image" + str(i), image)

            temp_message.append(image_base64string)

        message["boxes"] = temp_message
        if len(list) == 0:
            message["alarmType"] = "无告警"
        elif 0 < len(list) <= importantLowThreshold:
            message["alarmType"] = "一般"
        elif importantLowThreshold < len(list) <= urgentLowThreshold:
            message["alarmType"] = "重要"
        elif len(list) > urgentLowThreshold:
            message["alarmType"] = "紧急"

        # message["dst_image"]=dst_image_base64string
        message["message"] = "success"

    except Exception as e:
        app.logger.info("检测后数据处理异常:"+str(e))
        message["message"] = str(e)
        message["code"] = 500
        return message
    end_time = time.time()
    app.logger.debug("总运行时间："+ str(end_time - start_time))
    # cv2.waitKey()
    return message  # 响应json

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='5000') #参数不生效，原因未知 --host=0.0.0.0 --port=5000
