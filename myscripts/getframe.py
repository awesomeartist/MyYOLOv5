#!/usr/bin/env python3
# coding: utf-8
import os
import shutil
import cv2 as cv
from tqdm import tqdm
import glob
import numpy as np
 
 
def get_filelist(dir, Filelist):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            get_filelist(newDir, Filelist)
    return Filelist
 
def get_frame(path_name):
    save_dir = "piece"
    # save_dir = "G:\\体操\\picsPicked\\"

    cap = cv.VideoCapture(path_name)
 
    frames_num = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    isOpened = cap.isOpened()  ##判断视频是否打开
    print(isOpened)
 
    fps = cap.get(cv.CAP_PROP_FPS)  ##获取帧率
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))  ###获取宽度
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))  ###获取高度
    print(fps, width, height)
 
    np.random.seed(2020)
    list_num = np.array([x for x in range(frames_num)])
    np.random.shuffle(list_num)
    print(list_num)
    list_num = list_num[:int(frames_num*0.0025)]
 

    global piece_num
    i = 0
    # save_dir = os.path.join(save_dir, path_name.split("\\")[3])
    # if not os.path.exists(save_dir):
    # os.makedirs(save_dir)
 
    for n in range(frames_num):
        i = i + 1
        (flag, frame) = cap.read()
        if not flag:
            break
 
        # pic_name = dirName.split("\\")[-1].replace("afterPost","")
        # savdir = "G:\\D\\all\\" +pic_name
        # print(savdir +"_" +str(i) + ".jpg")
        if i in list_num:
 
            # dirName = "_".join(path_name.split("\\")[4:]).replace(".mp4", "").replace(".avi", "").replace(" ",
            #                                                                                               "_").replace(
            #     ".", "_")
            # dirNamenew = os.path.join(save_dir,dirName)+"_"+str(i) + ".jpg"
            # print(dirNamenew)

            number = str(piece_num)
            dirNamenew = './piece/'+ number.zfill(6) + ".jpg"
            print(dirNamenew)
            piece_num+=1
            # cv.imwrite(dirNamenew, frame, [cv.IMWRITE_JPEG_CHROMA_QUALITY, 100])  ##命名 图片 图片质量
            cv.imencode('.jpg', frame)[1].tofile(dirNamenew)
 
if __name__ == '__main__':
    # get all the abs path of videos
    list = get_filelist('c:/Users/liang/Desktop/myworks/project/dataset/mydata/origin', [])
    piece_num = 0
    for list_name in tqdm(list[0:len(list)]):
        # print(list_name)
        try:
            get_frame(list_name)
        except:
            continue
    # print(len(list))
 