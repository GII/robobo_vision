import cv2
from cv2 import aruco
import numpy as np

GENERAL_PATH = '/home/pol/Escritorio/TFG_2019-2020/Bibliotecas/'
ROBOBO_PATH = GENERAL_PATH+'robobo.py-master'
STREAMING_PATH = GENERAL_PATH+'robobo-python-video-stream-master/robobo_video'
import sys
sys.path.append(ROBOBO_PATH)
from Robobo import Robobo
from utils.Tag import Tag
sys.path.append(STREAMING_PATH)
from robobo_video import RoboboVideo

IP = '192.168.0.17'
rob = Robobo(IP)
rob.connect()
rob.moveTiltTo(80,70)
video = RoboboVideo(IP)
video.connect()



def Draw_Arucos(test, cord1,cord2,cord3,cord4,ids):
   
    print(cord1,cord2,cord3,cord4,ids)
    
    try:
        pts = np.array(((cord1['x'],cord1['y']),(cord2['x'],cord2['y']),(cord3['x'],cord3['y']),(cord4['x'],cord4['y'])))
        cv2.polylines(test,[pts],True,(0,255,0), thickness= 2)
        x,y = int(cord1['x']), int(cord1['y'])
        cv2.putText(test,str(ids),(x,y+20), 1,1, color= (0,0,255),thickness= 1)
                
    except  IndexError:
        pass

    return test


def tag_partition(tag):
    Id = tag.id 
    Coord_1 = tag.cor1 
    Coord_2 = tag.cor2
    Coord_3 = tag.cor3
    Coord_4 = tag.cor4
    Tvecs = tag.tvecs
    Rvecs = tag.rvecs

    return Id,Coord_1,Coord_2,Coord_3,Coord_4,Tvecs,Rvecs


def Light_Proof():
    dist = int(input('\nDISTANCIA DE LA MEDIDA: '))
    light = int(input('\nNIVEL DE LUZ DE LA MEDIDA: '))
    
    while True:
        frame = video.getImage()
        tag = rob.readTag()
        Id,Coord_1,Coord_2,Coord_3,Coord_4,Tvecs,Rvecs = tag_partition(tag)
        print(Rvecs)
        frame_p = Draw_Arucos(frame, Coord_1, Coord_2,Coord_3,Coord_4, Id)
        cv2.imshow('LIGHT_PROOF',frame_p)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('/home/pol/Escritorio/TFG_2019-2020/arUco/Pruebas_Luminosidad/100mm/Aruco_100_'+str(dist)+'_'+str(light)+'.jpg',frame)
            cv2.destroyAllWindows()
            break


def Angle_Proof():
    dist = int(input('\nDISTANCIA DE LA MEDIDA: '))
    angle = int(input('\nANGULO DE LA MEDIDA: '))
    angle = angle - 90
    while True:
        frame = video.getImage()
        tag = rob.readTag()
        Id,Coord_1,Coord_2,Coord_3,Coord_4,Tvecs,Rvecs = tag_partition(tag)
        print(Rvecs)
        frame_p = Draw_Arucos(frame, Coord_1, Coord_2,Coord_3,Coord_4, Id)
        cv2.imshow('ANGLE_PROOF',frame_p)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite('/home/pol/Escritorio/TFG_2019-2020/arUco/Pruebas_Angulo/100mm/Aruco_100_'+str(dist)+'_'+str(angle)+'.jpg',frame)
            cv2.destroyAllWindows()
            break


def Distance_Proof():
    dist = int(input('\nDISTANCIA DE LA MEDIDA: '))
    cont = 0
    while True:
        frame = video.getImage()
        tag = rob.readTag()
        Id,Coord_1,Coord_2,Coord_3,Coord_4,Tvecs,Rvecs = tag_partition(tag)
        cont+=1
        print(f'FRAME {cont} --> ID: {Id}\n')
 
        if Id == str(50):
            print('\nARUCO WITH ID 50 DETECTED\n')
            print(Tvecs['z']/10)
            cv2.imwrite('/home/pol/Escritorio/TFG_2019-2020/arUco/Pruebas_Distancia/100mm/Aruco_100_'+str(dist)+'_'+str(cont)+'.jpg',frame)
            frame_p = Draw_Arucos(frame, Coord_1, Coord_2,Coord_3,Coord_4, Id)
            cv2.imwrite('/home/pol/Escritorio/TFG_2019-2020/arUco/Pruebas_Distancia/100mm/Aruco_100_'+str(dist)+'_'+str(cont)+'_pintado.jpg',frame_p)
            break
        
        if cont > 2000:
            print('\nPROGRAM FINISHED\n')
            frame_p = Draw_Arucos(frame, Coord_1, Coord_2,Coord_3,Coord_4, Id)
            cv2.imwrite('/home/pol/Escritorio/TFG_2019-2020/arUco/Pruebas_Distancia/100mm/Aruco_100_'+str(dist)+'_'+str(cont)+'.jpg',frame)
            cv2.imwrite('/home/pol/Escritorio/TFG_2019-2020/arUco/Pruebas_Distancia/100mm/Aruco_100_'+str(dist)+'_'+str(cont)+'_pintado.jpg',frame_p)
            break
        

if __name__ == "__main__":
    #Se descomenta el que se vaya a usar dependiendo de la prueba.
    Distance_Proof()
    #Angle_Proof()
    #Light_Proof()
  