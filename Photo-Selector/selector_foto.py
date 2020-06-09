import cv2
import numpy as np
import pandas as pd
import time
import sys
sys.path.append('/home/pol/RAIZ/PROGRAMACIÓN/PYTHON/Robótica/ROBOBO/robobo.py-master')
from Robobo import Robobo
sys.path.append('/home/pol/RAIZ/UNIVERSIDAD/INGENIERÍA_EN_TECNOLOGÍAS_INDUSTRIALES/2019-2020/TFG/robobo-python-video-stream-master/robobo_video')
from robobo_video import RoboboVideo
""" sys.path.append('/home/pol/Escritorio/Ejemplos_FuncionesNuevas_Robobo/Mobilnet_SSD')
from  SSD_Mobilnet_OpenCV_ROBOBO import Mobilenet_SSD_OpenCV_FOTOS """
sys.path.append('/home/pol/RAIZ/UNIVERSIDAD/INGENIERÍA_EN_TECNOLOGÍAS_INDUSTRIALES/2019-2020/TFG/Object_detection_realtime')
from YoloV3_COCO import YoloV3
sys.path.append('/home/pol/Escritorio/TFG_2019-2020/Deteccion_de_objetos/Dataset TFG/Ejemplos_FuncionesNuevas_Robobo/Seleccion_imagenes')
from  SSD_Mobilnet_TF import Mobilenet_SSD_Tensorflow_Fotos


#Datos necesarios para la YOLO V3
ruta = '/home/pol/RAIZ/UNIVERSIDAD/INGENIERÍA_EN_TECNOLOGÍAS_INDUSTRIALES/2019-2020/TFG/Object_detection_realtime/'
pesos= ruta+'yolov3-tiny.weights'
config = ruta + 'yolov3-tiny.cfg'

#Datos necesarios para la Mobilenet V2
rutas='/home/pol/Escritorio/Ejemplos_FuncionesNuevas_Robobo/Mobilnet_SSD/'
pesoss= rutas+'frozen_inference_graph.pb'
configg = rutas + 'graph.pbtxt' 

#Datos necesarios para V3 que corre en local
ruta = '/home/pol/Escritorio/TFG_2019-2020/Deteccion_de_objetos/Dataset TFG/Ejemplos_FuncionesNuevas_Robobo/Seleccion_imagenes/'
pesos3= ruta + 'frozen_inference_graph.pb'


class Selector_fotos:

    def __init__(self, tipo,n,IP='10.113.36.145'):
        self.tipo = tipo
        self.n = n
        self.IP = IP
        self.num_fotos = 10
        self.tilt=90
        
    def streaming_prueba(self):
        robobo = Robobo(self.IP)
        robobo.connect()
        robobo.moveTiltTo(self.tilt, 30)
        self.video = RoboboVideo(self.IP)
        self.video.connect()
        while True:
            self.img = self.video.getImage()
            cv2.imshow('streaming Prueba', self.img)
            if cv2.waitKey(1) & 0XFF == ord('s'):
                break
        self.video.disconnect()
        cv2.destroyAllWindows()


    def guarda_Archivo(self,tomas,num,ruta):
        f = open (ruta + 'Datos_toma_'+str(self.n)+'/Datos_foto'+str(num)+'.txt','w')
        for elemento in tomas:
            f.write('%s \n' % elemento)
        f.close()

    def leer_txt(self, ruta, num=8):

        data = pd.read_csv(ruta + 'Datos_toma_'+str(self.n)+'/Datos_foto'+str(num)+'.txt', header = None)
        return data[1][0], data[2][0], data[3][0], data[4][0]

           
    def guardar_csv(self,confianza,x,y,area):

        datas = {'confidence': confianza,
                'X': x,
                'Y': y,
                'Area': area}

        df = pd.DataFrame(datas, columns=['confidence', 'X', 'Y', 'Area'])
        
        df.to_csv('datos.csv')

    def recolector_datos(self,ruta = None,mostrar_img=False):
        
        for i in range(self.num_fotos):
            self.frame=cv2.imread(ruta +'Fotos_toma_'+str(self.n)+'/foto'+str(35)+'.jpg')
            tipo = 'yolo'
            tipo2 = 'mobilenet'
            tipo3='mobilenet3'
            if self.tipo == tipo:
                red = YoloV3(pesos, config)
                pintada, datos = red.datos_fotos(self.frame)
            elif self.tipo == tipo2:
                pintada, datos = Mobilenet_SSD_OpenCV_FOTOS(pesoss, configg, self.frame)
            elif self.tipo == tipo3:
                red=Mobilenet_SSD_Tensorflow_Fotos(pesos3,self.frame)
                pintada,datos=red.tensorflow_session()
            
            cv2.imwrite(ruta +'Fotos_toma_'+str(self.n)+'/foto_pintada'+str(35)+'.jpg',pintada)
            self.guarda_Archivo(datos, 35,ruta)
            if mostrar_img:
                while True:
                    cv2.imshow(f'Imagen pintada {i+1}', pintada)
                    if cv2.waitKey(1) & 0xFF == ord("s"):
                        break
                cv2.destroyAllWindows()
            
            
    def sacador_fotos(self,ruta_guardado=None):
        robobo = Robobo(self.IP)
        robobo.connect()
        robobo.moveTiltTo(self.tilt, 30)
        self.video = RoboboVideo(self.IP)
        self.video.connect()
        print('sacando fotos')
        robobo.wait(5)
        for i in range(self.num_fotos):
            self.frame = self.video.getImage()
            #ruta_guardado = '/home/pol/Escritorio/Ejemplos_FuncionesNuevas_Robobo/Seleccion_imagenes/'
            cv2.imwrite(ruta_guardado + 'Fotos_toma_'+str(self.n)+'/foto'+str(i+1)+'.jpg',self.frame)
        self.video.disconnect()
        print('\nFotos sacadas\n')

ruta = '/media/pol/kk/Dataset_TFG/DATASET_TFG_Mobilenet/Nuevas_tomas/ocluidas_combinacion2_5_tomas/'



confianza = []
x = []
y = []
area = []

for num in range(5,7):
    print('Prueba '+str(num+1))  
    pruebas = Selector_fotos('mobilenet3', num+1 ,IP='192.168.0.17')
    pruebas.tilt = 100
    pruebas.num_fotos=10
    pruebas.sacador_fotos(ruta_guardado= ruta)
    print('\n\nprepara la siguiente prueba\n\n')  
    time.sleep(10)
    pruebas.recolector_datos(ruta) 
    #confi, xs, ys, areas = pruebas.leer_txt(ruta)
    #confianza.append(confi)
    #x.append(xs)
    #y.append(ys)
    #area.append(areas)
#pruebas.streaming_prueba()
#pruebas.guardar_csv(confianza,x,y,area)
'''

score,area_y=pruebas.lector_datos(ruta,'bottle')
print(area_y)
print(len(area_y))
area_y=[ 3395 for j in range(60)]
x=[i+1 for i in range(60)]
area=[3309.73407062652, 3340.7140845141566, 3340.7140845141566, 3340.7140845141566, 3340.7140845141566 ,  3340.7140845141566 ,  3340.7140845141566,  3340.7140845141566, 3340.7140845141566 ,  3340.7140845141566, 3340.7140845141566,  3340.7140845141566,  3337.6871450527688,  3337.6871450527688,  3337.6871450527688,  3337.6871450527688,  3337.6871450527688,  3337.6871450527688, 3337.6871450527688, 3337.687145052768, 3337.6871450527688, 3337.6871450527688, 3337.6871450527688,  3337.6871450527688, 3325.141121802194 ,  3325.141121802194,  3325.141121802194,  3325.14112180219, 3325.141121802194,  3325.141121802194,  3325.141121802194,  3325.141121802194,  3325.141121802194 ,  3325.141121802194 ,  3325.141121802194,  3325.141121802194, 3325.141121802194,  3325.141121802194,  3325.141121802194,  3325.141121802194,  3325.141121802194,  3325.141121802194,  3325.141121802194,  3325.141121802194, 3343.7106676767144,  3343.7106676767144,  3343.7106676767144,  3343.7106676767144,  3343.7106676767144,  3343.7106676767144,  3343.7106676767144,  3343.7106676767144,  3343.7106676767144,  3343.710667676714, 3343.710667676714 ,  3343.7106676767144,  3343.71066767671, 3343.71066,3343.71066, 3343.71066]

plt.plot(x, area, 'r',label='Mobilenet')
plt.plot(x, area, 'ro')
plt.plot(x, area_y, 'b',label='Yolo')
plt.plot(x, area_y, 'bo')
plt.grid()
plt.title('Area Variation')
plt.xlabel('Measures')
plt.ylabel('Area')
plt.legend()
plt.ylim(2000,3500)
plt.show()
#pruebas.streaming_prueba()

'''