import cv2
import numpy as np

GENERAL_PATH = '/home/pol/Escritorio/TFG_2019-2020/Bibliotecas/'
ROBOBO_PATH = GENERAL_PATH+'robobo.py-master'
STREAMING_PATH = GENERAL_PATH+'robobo-python-video-stream-master/robobo_video'
MOBILENETV2_PATH = '/home/pol/RAIZ/UNIVERSIDAD/INGENIERÍA_EN_TECNOLOGÍAS_INDUSTRIALES/2019-2020/TFG/Mobilenet_SSD_COCO'
YOLOV3_PATH  = '/home/pol/RAIZ/UNIVERSIDAD/INGENIERÍA_EN_TECNOLOGÍAS_INDUSTRIALES/2019-2020/TFG/Object_detection_realtime'

import sys
sys.path.append(ROBOBO_PATH)
from Robobo import Robobo
from utils.Tag import Tag
sys.path.append(STREAMING_PATH)
from robobo_video import RoboboVideo
sys.path.append(MOBILENETV2_PATH)
from SSD_Mobilnet_OpenCV_ROBOBO import Mobilenet_SSD_OpenCV_ROBOBO
sys.path.append(YOLOV3_PATH)
from YoloV3_COCO import YoloV3

#Datos necesarios para la YOLO V3
ruta = '/home/pol/RAIZ/UNIVERSIDAD/INGENIERÍA_EN_TECNOLOGÍAS_INDUSTRIALES/2019-2020/TFG/Object_detection_realtime/'
pesos= ruta+'yolov3-tiny.weights'
config = ruta + 'yolov3-tiny.cfg'

#Datos necesarios para la Mobilenet V2
pesoss= MOBILENETV2_PATH+'/frozen_inference_graph.pb'
configg = MOBILENETV2_PATH+'/graph.pbtxt'

IP = '192.168.0.17'
rob = Robobo(IP)
rob.connect()
rob.moveTiltTo(100,70)
video = RoboboVideo(IP)
video.connect()

def DrawObject(img, obj):
        label = obj.label
        scores = obj.confidence
        x=obj.x 
        y=obj.y 
        h=obj.height 
        w=obj.width 
        cv2.rectangle(img, (x-int(w/2), y-int(h/2)), (x + int(w/2), y + int(h/2)), (0,255,0) , 2)
        cv2.rectangle(img, (x-int(w/2), y-int(h/2) ), (x+int(w/2), y - int(h/2)+30), (0,255,0), -1)
        cv2.putText(img, label + " " + str(round(scores*100, 2))+"%", (x-int(w/2), y - int(h/2) +20 ), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,255,255), 1)
        
        return img


def Analize_Frames():
    cont=0
    while True:
        frame = video.getImage()
        red_Yolo = YoloV3(pesos,config)
        _ ,datos_yolo = red_Yolo.datos_fotos(frame)
        print(f'\nYOLO V3: {datos_yolo}\n')
        _ ,datos_mobilenet = Mobilenet_SSD_OpenCV_ROBOBO(frame, pesoss, configg)
        print(f'\nMOBILENET V2: {datos_mobilenet}\n')
        obj = rob.readDetectedObject()
        frame_v3_pintado = DrawObject(frame,obj)
        print(f'\nMOBILENET V3: {obj.label} -> {obj.confidence }\n')
        '''
        cont+=1
        if cont >=1:
            break
        '''
        cv2.imshow('Smartphone Camera', frame_v3_pintado)
        cv2.imwrite('/home/pol/Escritorio/botella.jpg',frame_v3_pintado)
        if cv2.waitKey(1) & 0xFF == ord('q'):

            video.disconnect()
            cv2.destroyAllWindows()
            print('\nDONE\n')
            rob.sayText('Fotos sacadas. Siguiente objeto')
        


if __name__ == "__main__":

    Analize_Frames()
    