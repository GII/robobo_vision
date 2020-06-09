
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from simple_pid import PID
import sys
#CAMBIAR RUTAS POR LAS NUESTRAS
GENERAL_PATH = '/home/pol/Escritorio/TFG_2019-2020/Bibliotecas/'
ROBOBO_PATH = GENERAL_PATH+'robobo.py-master'
STREAMING_PATH = GENERAL_PATH+'robobo-python-video-stream-master/robobo_video'
#Cambiar IP por el de nuestro dispositivo
IP = '192.168.43.36'

sys.path.append(ROBOBO_PATH)
from Robobo import Robobo

sys.path.append(STREAMING_PATH)
from robobo_video import RoboboVideo


def draw_line_pro(coeffs1, coeffs2, M, frame, h=130):
    """
    Draws the lines with coeffs1 and coeffs2 on a mask that is warped using the transformation matrix M, then its added to the frame
    
    :param coeffs1: coefficients of line 1
    :param coeffs2: coefficients of line 2
    :param M: transformation matrix
    :param frame: frame

    :return: frame with the lines drawn
    """
    # obten altura y ancho de la imagen
    try:
        height, width, _ = frame.shape
        # Mascara para dibujar las lineas
        mask = np.zeros_like(frame)

        # Crea un Array hasta heigh-1
        plot_y = np.linspace(0, height - 1, height)

        
        left_x = coeffs1['a'] * plot_y ** 2 \
                + coeffs1['b'] * plot_y \
                + coeffs1['c']
        right_x = coeffs2['a'] * plot_y ** 2 + \
                coeffs2['b'] * plot_y + \
                coeffs2['c']

        x1 = 0
        y1 = height- h
        x2 = width
        y2 = height- h
        
        #if right_x !=  mask:
        cv2.polylines(mask, [np.int32(np.stack((right_x, plot_y), axis=1))], False, (0, 0, 255), 20)
        #if left_x != mask:
            # Draw the lines (one red, one blue)
        cv2.polylines(mask, [np.int32(np.stack((left_x,plot_y), axis=1))], False, (255, 0, 0), 20)
            
    
        # Warpea la perspectiva
        mask = cv2.warpPerspective(mask, np.float32(M), (width, height))  # Warp back to original image space

        # Añade las lineas a la imagen original
        img = cv2.addWeighted(frame, 1., mask, 0.5, 0)
        x_r, x_l = right_x[height - h], left_x[height -h]
        
        
        #Calcula el punto medio 
        x_medio = ((x_r - x_l)/2) + x_l

    
        #Lineas horizontales de corto
        #cv2.line(img, (x1,y1), (x2,y2), (102,255,102), 2)

        #Direction Line
        cv2.line(img, (int(width/2), height- 1), (int(x_medio),height-h), (255,102,178), 3)

    except ValueError:
        pass

    return img, x_medio

def round_unit(num):
    """
    Round any decimal number to the nearest unit

    :param num: Float number 

    :return: Int number rounded to unity
    """
    try:
        neg= False
        if num < 0:
            num=abs(num)
            neg = True

        if (num - int(num)) > 0.5:
            num=int(num)+1
        else:
            num = int(num)

        if neg:
            num= -num
    except ValueError:
        pass

    return num   


def AutoTunning(kp,ki,kd,Input,mitad):
    #Auto-Tunning
    pid = PID(kp,ki,kd, setpoint= mitad)
    #pid.output_limits = (0, vel + 15)
    pid.sample_time = 0.01

    #computa nuevo pid
    control = pid(Input)
    kp, ki, kd = pid.components
    print(kp,ki,kd)
    return kp,ki,kd

############################################### Comienzo del Programa  #########################################################
rob = Robobo(IP)
rob.connect()
rob.moveTiltTo(100, 70)
#rob.movePanTo(0,70)
video = RoboboVideo(IP)
video.connect()
rob.setLaneColorInversion(False)
draw = False

#Elige entre la ejecución con o sin streaming
while True: 
    resp = input('\nQuieres usar el streaming (Ralentiza el control) (s/n): ')

    if resp == 's':
        draw = True
        break
    elif resp == 'n':
        draw = False
        break
    else:
        print("\nIntroduce 's' o 'n'\n")
print("\nRecuerda pulsar la tecla 'q' para parar el script en cualquier momento\n")
time.sleep(2)
print("\n               CONTROL EN MARCHA         \n")
time.sleep(1)

frame = video.getImage()
_, width, _ = frame.shape

#PARAMETROS
vel = 10
rang = 10 
i = 0
acum = 0                       #intervalo de  +- pixeles desde el centro del carril que se continua considerando centro
lista_ep=[]
lista_incr=[]
kp = 0.1                   #Ajustar parámetros kp y ki
ki = 0.97
kd = 0
mitad = width/2             #Valor fijo. Coordenada X de la mitad de la imagen

while True:

    frame = video.getImage()
    if frame is None:
        continue
    
    #Obtengo los coeficientes de los carriles detectados y su matriz de tranformación
    obj = rob.readLanePro()
    coeffs1 = obj.coeffs1
    coeffs2 = obj.coeffs2
    M = obj.minv
    #Pinto si quiero lo que devuelve el método y calculo el punto medio del carril
    frame, x_medio = draw_line_pro(coeffs1, coeffs2, M, frame)
    Input = x_medio

    #kp,ki,kd = AutoTunning(kp,ki,kd,Input,mitad)  #Se puede usar un Auto Tunning para ajustar los parámetros PID
    
    #Proporcional−Integral
                          # i vale inicialmente 1 y en cada iteración del bucle while aumenta en uno
    ep=(Input - mitad)                  #calculo del error
    lista_ep.append(ep)
    incr = abs(ep)*kp            #control solo proporcional
    lista_incr.append(incr)      #Guardo los valores de incremento y error en el tiempo
        
    if i >=1:                            #A partir de a segunda iteración se le añade el control integral
        incr = ep - lista_ep[-1] * ki + lista_incr[-1]
        lista_incr.append(incr)
        
    #Cada vez que se han guardado más de 50 elementos en la lista, elimino el valor de la posición 0 para no sufrir desbordamiento
    if i >  50:               
        lista_ep.pop(0)
        lista_incr.pop(0)
        
    #Limitador del valor de salida. 
    #Todo valor de incremento negativo se hace cero
    if incr < 0:
        incr = 0
    #se permite que el máximo incremento de velocidad  para una velocidad dada sea siempre menor que 25
    elif incr >15:
        incr= 15
            
    #redondea a la unidad más cercana el valor del incremento ya que el método moveWheels solo acepta enteros
    if incr < 1 and incr > 0:
        incr= round_unit(incr)
    else:
        incr= round_unit(incr)

    # Se muestra por pantalla la velocidad del Robobo en su rueda de giro
    text = vel+incr 
    vel_left , vel_right = vel, vel
    
    if coeffs1['a'] == 0 and coeffs2['a'] == 0:
        acum +=1
        if acum > 70:
            print('\nCarriles perdidos de vista\n')
            rob.stopMotors()
            break
         
        
    #CONTROL DE VELOCIDAD

    if coeffs1['a'] == 0:
        #acum = 0
        print(coeffs1)
        print('\ncarril izquierdo perdido\n')
        print('\nDebo girar a la derecha\n')
        rob.moveWheels(vel , vel + 15)
        
        
    if coeffs2['a'] == 0:
        #acum = 0
        print(coeffs2)
        print('\ncarril derecho perdido\n')
        print('\nDebo girar a la izquierda\n')
        rob.moveWheels(vel + 15, vel  )
     
    if coeffs1['a'] != 0 and coeffs2['a'] != 0:
        print('Carriles detectados')
        #acum = 0
        rob.moveWheels(vel,vel)
    
        if ep > rang:
            rob.moveWheels(vel+incr,vel)
            print('TURN TO THE LEFT\n')
            print(f"\nRight wheel speed --> {text}\n")
            vel_right = vel +incr
        elif ep < -rang:
            rob.moveWheels(vel,vel+incr)
            print('TURN TO THE RIGHT\n')
            print(f"\nLeft wheel speed --> {text}\n")
            vel_left = vel + incr
        else:
            rob.moveWheels(vel,vel)
            print('\nSTAY ON COURSE\n')
    
              
    i+=1
    cv2.namedWindow('Running Control')
    if draw:
        #Muestra por pantalla lo que ve el método
        cv2.putText(frame,f"Right wheel speed --> {vel_right}",(5,20),fontFace= cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.7, color = (0,0,0))
        cv2.putText(frame,f"Left wheel speed  --> {vel_left}",(5,40),fontFace= cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.7, color = (0,0,0))
        cv2.imshow('Smarphone Camera', frame)
        cv2.imwrite('/home/pol/Escritorio/curva_vel_top'+str(vel)+'.jpg', frame)
           
    #Al pulsar 'q' finaliza el script       
    if cv2.waitKey(1) & 0XFF == ord('q'):
            rob.stopMotors()
            video.disconnect()
            cv2.destroyAllWindows()
            break

rob.stopMotors()
video.disconnect()
cv2.destroyAllWindows()
print('\nPROGRAM FINISHED\n')
