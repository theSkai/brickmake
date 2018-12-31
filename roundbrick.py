import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as ima
import tkinter

# 返回 (色阶数组，左边界，右边界)
def sejie(pic):
    S = np.zeros(256)
    row, col = pic.shape[0], pic.shape[1]
    for i in range(row):
        for j in range(col):
            gray = pic[i, j]
            S[gray] = S[gray] + 1
    ts_co = 60  # 阈值比例
    ts = np.max(S) / ts_co  # 阈值
    gray_low = 0  # 左边界
    gray_high = 0  # 右边界
    for i in range(0, 256):
        if S[i] > ts:
            gray_low = i
            break
    for i in range(0, 256):
        if S[255 - i] > ts:
            gray_high = 255 - i
            break
    return(S, gray_low, gray_high)
# 扩展
def kuozhan(pic, low, high):#low:最低值，high:最高值
    imgshape = pic.shape
    row, col = imgshape[0], imgshape[1]
    newpic = np.zeros(imgshape)
    for i in range(row):
        for j in range(col):
            if pic[i][j] < low:
                newpic[i][j] = 0
            elif pic[i][j] > high:
                newpic[i][j] = 255
            else:
                newpic[i][j] = (pic[i][j] - low) * 255 / (high - low)
    return newpic.astype(np.int)
# RGB to 灰度
def rgb2gray(rgb):
    temp = np.dot(rgb[...,:3], [0.289, 0.577, 0.134])
    return temp.astype(np.int)
# 分块，返回四维(n_height, n_width, matrix_kuai[][])
def convert(gray, nh):
    size = gray.shape
    height = size[0]
    width = size[1]
    innerh = int(height / nh)
    nw = int(width / innerh)
    #print(innerh)

    # 自己取整了

    piece = np.empty([nh, nw, innerh, innerh], dtype=int)
    #print(len(piece))

    for i in range(nh):
        for j in range(nw):
            for k in range(innerh):
                for l in range(innerh):
                    piece[i][j][k][l] = gray[i*innerh + k][j*innerh + l]

    #print(piece)

    return piece

# mapping 返回二维数组
def mapping(matrix, pic, n_height, n_width, tile_width = 32):
    """接口函数"""
    temp = np.zeros((n_height,n_width))
    #max, min = getmaxmin(matrix, pic, n_height, n_width, tile_width)
    for h in range(n_height):
        for w in range(n_width):
            temp[h][w] = map1(matrix[h][w], pic, tile_width)
    return temp;
def getmaxmin(matrix, pic, n_height, n_width, tile_width = 32):
    max = 0
    min = 255
    for h in range(n_height):
        for w in range(n_width):
            for hh in range(tile_width):
                for ww in range(tile_width):
                    if matrix[h][w][hh][ww] > max:
                        max = matrix[h][w][hh][ww]
                    if matrix[h][w][hh][ww] < min:
                        min = matrix[h][w][hh][ww]
    return max, min
def map1(mat, pic, tile_width = 32):
    '''输出整数0-9，0代表最浅的砖，9代表最深的'''
    eigen = 0.0;
    for h in range(tile_width):
        for w in range(tile_width):
            eigen += mat[h][w]
    mean = eigen/tile_width/tile_width
    eigen = 9 - int((mean - 1)/25.6)
    return eigen
def map2(mat, pic, tile_width = 32, max = 255, min = 0):
    '''另一种方式'''
    length = max - min
    eigen = 0.0;
    for h in range(tile_width):
        for w in range(tile_width):
            eigen += mat[h][w]
    mean = eigen/tile_width/tile_width
    eigen = 9 - int((mean - 1 - min)*10/length)
    return eigen

# brickmanufact 返回图片(33*nh-1, 33*nw-1)
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
def drawroundbrick(xsize,x,outcolor,incolor):
    littlebrick = np.empty([xsize, xsize], dtype=int)
    for i in range (0,xsize):
        for j in range(0,xsize):
            littlebrick[i][j]=outcolor
    if x==0:
        for i in range(0, xsize):
            for j in range(0, xsize):
                if (i - 32) * (i - 32) + (j - 32) * (j - 32) <=25:
                    littlebrick[i][j] = incolor
    elif x==1:
        for i in range(0,xsize):
            for j in range(0, xsize):
                if (i-32)*(i-32)+(j-32)*(j-32)<=64:
                    littlebrick[i][j] = incolor
    elif x==2:
        for i in range(0,xsize):
            for j in range(0, xsize):
                if (i-32)*(i-32)+(j-32)*(j-32)<=64:
                    littlebrick[i][j] = incolor
    elif x==3:
        for i in range(0,xsize):
            for j in range(0, xsize):
                if (i-32)*(i-32)+(j-32)*(j-32)<=121:
                    littlebrick[i][j] = incolor
    elif x==4:
        for i in range(0,xsize):
            for j in range(0, xsize):
                if (i-32)*(i-32)+(j-32)*(j-32)<=196:
                    littlebrick[i][j] = incolor
    elif x==5:
        for i in range(0,xsize):
            for j in range(0, xsize):
                if (i-32)*(i-32)+(j-32)*(j-32)<=441:
                    littlebrick[i][j] = incolor
    elif x==6:
        for i in range(0,xsize):
            for j in range(0, xsize):
                if((i+j)>=34)&((i+j)<=94)&((j-i)<=30)&((i-j)<=30):
                    littlebrick[i][j] = incolor
                if (i-32)*(i-32)+(j-32)*(j-32)<=441:
                    littlebrick[i][j] = incolor
    elif x==7:
        for i in range(0,xsize):
            for j in range(0, xsize):
                if((i+j)>=32)&((i+j)<=96)&((j-i)<=32)&((i-j)<=32):
                    littlebrick[i][j] = incolor
    elif x==8:
        for i in range(0,xsize):
            for j in range(0, xsize):
                if((i+j)>=20)&((i+j)<=108)&((j-i)<=44)&((i-j)<=44):
                    littlebrick[i][j] = incolor
    elif x==9:
        for i in range(0,xsize):
            for j in range(0, xsize):
                if((i+j)>=10)&((i+j)<=118)&((j-i)<=54)&((i-j)<=54):
                    littlebrick[i][j] = incolor
    else:
        print ('false')
    return littlebrick
def drawcolor(xsize,x):
    littlebrick = np.empty([xsize, xsize], dtype=int)
    for i in range (0,xsize):
        for j in range(0,xsize):
            littlebrick[i][j]=225
    if x==0:
        for i in range (0,xsize):
            print(1)
    elif x==1:
        for i in range(0,xsize):
            for j in range(0, xsize):
                littlebrick[i][j] = 200
    elif x==2:
        for i in range(0,xsize):
            for j in range(0, xsize):
                littlebrick[i][j] = 175
    elif x==3:
        for i in range(0,xsize):
            for j in range(0, xsize):
                littlebrick[i][j] = 150
    elif x==4:
        for i in range(0,xsize):
            for j in range(0, xsize):
                littlebrick[i][j] = 125
    elif x==5:
        for i in range(0,xsize):
            for j in range(0, xsize):
                littlebrick[i][j] = 80
    elif x==6:
        for i in range(0,xsize):
            for j in range(0, xsize):
                littlebrick[i][j] = 40
    elif x==7:
        for i in range(0,xsize):
            for j in range(0, xsize):
                littlebrick[i][j] = 25
    elif x==8:
        for i in range(0,xsize):
            for j in range(0, xsize):
                littlebrick[i][j] = 10
    elif x==9:
        for i in range(0,xsize):
            for j in range(0, xsize):
                littlebrick[i][j] =0
    else:
        print ('false')
    return littlebrick
def append(finalpic,colormatrix,i,j,outcolor,incolor,xsize):#i行j列砖头上色
    typecolor=colormatrix[i][j]
    unitmatrix=drawcolor(xsize,typecolor)
    for hang in range(0,xsize):
        for lie in range(0,xsize):
            finalpic[(xsize+1)*i+hang][(xsize+1)*j+lie]=unitmatrix[hang][lie]
    return finalpic
def brickmanufact(colormatrix,jointwidth,outcolor,incolor,xsize):#最终图片生成
    nh=len(colormatrix)
    nw=len(colormatrix[0])
    finalpic=np.empty([nh*(xsize+1)-1,nw*(xsize+1)-1],dtype=int)
    for i in range(0,nh*(xsize+1)-1):
        for j in range(0,nw*(xsize+1)-1):
            finalpic[i][j]=100
    #全部刷上墙缝颜色
    for i in range(0,nh):
        for j in range(0,nw):
            finalpic=append(finalpic,colormatrix,i,j,outcolor,incolor,xsize)
    return finalpic#一个个刷砖块

src = "222.jpg" #文件名
n_height = 60 #高度的砖块数
jointwidth=1 #墙缝宽度
outcolor=250
incolor=50
xsize=65#x代表了所画砖块的尺寸


img = ima.imread(src)
img = rgb2gray(img)
img_sejie = sejie(img)
#img = kuozhan(img, img_sejie[1], img_sejie[2])
matrix = convert(img, n_height)
n_width = matrix.shape[1]
len_brick = matrix.shape[2]
index = mapping(matrix, 0, n_height, n_width, len_brick)
finalmap = brickmanufact(index,jointwidth,outcolor,incolor,xsize)#colormatrix是待输入的上色方案
#finalmap[0][0] = 0;
#finalmap[0][1] = 255;

#print(img.shape)
#print(index.shape)
#print(index)
#print(index[49])
plt.imshow(finalmap, cmap="gray")
plt.savefig('./teacherres.jpg')
name=""
cv2.imwrite("feifei.jpg",finalmap)
plt.show()