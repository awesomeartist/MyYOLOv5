import os
import shutil

def split_file(src, dst1, dst2):
    '''
    function: 将文件中不同后缀的文件分开到不同文件夹
    example: 区分jpg和png图像
    src:str(filefolder)
    dst:str(filefolder)
    '''
    #区分jpg和png
    jpg = []
    png = []
    for f in os.listdir(src):
        if f.endswith('.jpg'):
            jpg.append(f)
        elif f.endswith('.xml'):
            png.append(f)
    #创建目标文件夹
    if not os.path.isdir(dst1):
        os.mkdir(dst1)
    if not os.path.isdir(dst2):
        os.mkdir(dst2)
    #拷贝文件到目标文件夹
    for j in jpg:
        _jpg = os.path.join(src, j)
        shutil.copy(_jpg, dst1)
    for p in png:
        _png = os.path.join(src, p)
        shutil.copy(_png, dst2)
 #example
if __name__ == '__main__':

    base_filename = '.'
    src = os.path.join(base_filename, 'data')
    dst1 = os.path.join(base_filename, 'jpg_file')
    dst2 = os.path.join(base_filename, 'xml_file')
    split_file(src, dst1, dst2)



