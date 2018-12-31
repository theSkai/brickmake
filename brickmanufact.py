import cv2
import matplotlib.image as ima
import matplotlib.pyplot as plt
import numpy as np
def drawbrick(x,outcolor,incolor):
    littlebrick = np.empty([32, 32], dtype=int)
    for i in range (0,32):
        for j in range(0,32):
            littlebrick[i][j]=outcolor
    if x==0:
        for i in range (0,0):
            print(1)
    elif x==1:
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
#10种砖块小矩阵生成
def append(finalpic,colormatrix,i,j,outcolor,incolor):#i行j列砖头上色
    typecolor=colormatrix[i][j]
    unitmatrix=drawbrick(typecolor,outcolor,incolor)
    for hang in range(0,32):
        for lie in range(0,32):
            finalpic[33*i-33+hang][33*j-33+lie]=unitmatrix[hang][lie]
    return finalpic

def brickmanufact(colormatrix,jointwidth,outcolor,incolor):#最终图片生成
    nh=len(colormatrix)
    nw=len(colormatrix[0])
    finalpic=np.empty([nh*33-1,nw*33-1],dtype=int)
    for i in range(0,nh*33-1):
        for j in range(0,nw*33-1):
            finalpic[i][j]=255
    #全部刷上墙缝颜色
    for i in range(1,nh):
        for j in range(1,nw):
            finalpic=append(finalpic,colormatrix,i,j,outcolor,incolor)
    return finalpic#一个个刷砖块



x=3
y=4
jointwidth=1#墙缝宽度
outcolor=255
incolor=0
colormatrix=np.zeros([x,y],dtype=int)
finalmap=brickmanufact(colormatrix,jointwidth,outcolor,incolor)#colormatrix是待输入的上色方案