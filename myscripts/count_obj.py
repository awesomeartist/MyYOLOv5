# -*- coding:utf-8 -*-
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt

def parse_obj(xml_path, filename):
    tree = ET.parse(xml_path + filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        objects.append(obj_struct)
    return objects

def read_image(image_path, filename):
    im = Image.open(image_path + filename)
    W = im.size[0]
    H = im.size[1]
    area = W * H
    im_info = [W, H, area]
    return im_info

if __name__ == '__main__':
    xml_path = r'./Annotations/'
    filenamess = os.listdir(xml_path)
    filenames = []
    for name in filenamess:
        name = name.replace('.xml', '')
        filenames.append(name)
    recs = {}
    obs_shape = {}
    classnames = []
    num_objs = {}
    obj_avg = {}
    for i, name in enumerate(filenames):
        recs[name] = parse_obj(xml_path, name + '.xml')
    for name in filenames:
        for object in recs[name]:
            # if object['name'] == 'do(onheight)':
            #     print(name)
            if object['name'] not in num_objs.keys():
                num_objs[object['name']] = 1

            else:
                num_objs[object['name']] += 1
            if object['name'] not in classnames:
                classnames.append(object['name'])
    for name in classnames:
        print('{}:{}个'.format(name, num_objs[name]))
    print('信息统计算完毕。')

    plot_image = True
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(classnames)), list(num_objs.values()), align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        # plt.xticks(range(len(classnames)), classnames)
        plt.xticks(range(len(classnames)), ['1','2','3','4','5','6','7','8','9'])
        # 在柱状图上添加数值标签
        for i, v in enumerate(num_objs.values()):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('object class')
        # 设置y坐标
        plt.ylabel('number of object')
        # 设置柱状图的标题
        plt.title('object class distribution')
        plt.show()
