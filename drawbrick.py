import cv2
import matplotlib.image as ima
import matplotlib.pyplot as plt
import numpy as np

def drawbrick(x,outcolor,incolor):
    littlebrick = np.empty([32, 32], dtype=int)
    for i in range (0,32):
        for j in range(0,32):
            littlebrick[i][j]=outcolor
    if x==1:
        for i in range(15,17):
            for j in range(15, 17):
                littlebrick[i][j] = incolor
    elif x==2:
        for i in range(14,18):
            for j in range(14, 18):
                littlebrick[i][j] = incolor
    elif x==3:
        for i in range(12,20):
            for j in range(12, 20):
                littlebrick[i][j] = incolor
    elif x==4:
        for i in range(10,22):
            for j in range(10, 22):
                littlebrick[i][j] = incolor
    elif x==5:
        for i in range(8,24):
            for j in range(8, 24):
                littlebrick[i][j] = incolor
    elif x==6:
        for i in range(6,26):
            for j in range(6, 26):
                littlebrick[i][j] = incolor
    elif x==7:
        for i in range(4,28):
            for j in range(4, 28):
                littlebrick[i][j] = incolor
    elif x==8:
        for i in range(2,30):
            for j in range(2, 30):
                littlebrick[i][j] = incolor
    elif x==9:
        for i in range(0,32):
            for j in range(0, 32):
                littlebrick[i][j] = incolor
    else:
        print ('false')
    return littlebrick

#main
x=7
print (drawbrick(x,255,0))#输入外色、内色


