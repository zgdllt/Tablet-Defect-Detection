#使用前注意更改roi范围,面积阈值,轮廓扩张大小
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
seq=1 #起始序号s
image_dir="tablets" #图片文件夹
RED=(0,0,255)
BLUE=(255,0,0)
def get_value(img,x,y,c):
    sum=0
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):
            sum+=int(img[j,i,c]) # In opencv image,x and y are reversed
    return sum//9
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
img_name=[f for f in os.listdir(image_dir) if f.endswith('.bmp')]
for var in img_name: #图片个数
    img=cv2.imread(image_dir+"\\"+var) #命名
    if img is None:
        continue
    img=img[0:2048,800:1400] #ROI设定,前一参数为纵坐标，后为横
    # Convert the image into a grayscale image using only the R and G channels.
    # In OpenCV, the channel order is BGR, so index 2: R, index 1: G.
    gray = ((img[:, :, 1].astype(np.float32) + img[:, :, 2].astype(np.float32)) / 2).astype(np.uint8)
    ret1, thresh=cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
    cv_show("thresh",thresh)
    contours,hierarchy=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    draw_img=img.copy()
    chk=[]
    for cnt in contours:
        draw_img=cv2.drawContours(draw_img,cnt,-1,RED,2)
        if cv2.contourArea(cnt)>500: #面积阈值
            x,y,w,h=cv2.boundingRect(cnt)
            chk.append([x,y,w,h,cnt])
    plt.imshow(draw_img)
    for i in chk:
        testcontour=np.array([np.array([i[0],i[1]]),np.array([i[0]+i[2],i[1]]),np.array([i[0]+i[2],i[1]+i[3]]),np.array([i[0],i[1]+i[3]])])
        temp=i
        template=img[i[1]-5:i[1]+155,i[0]-5:i[0]+155] #轮廓扩张
        if template is None:
            continue
        try:
            stat=cv2.imwrite("data"+str(seq)+".bmp",template) #输出命名
        except:
            continue
        seq+=1