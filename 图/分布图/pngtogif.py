import matplotlib.pyplot as plt
import imageio,os
from PIL import Image
GIF=[]
filepath="E:\workspace\广东\图\分布图\预测"#文件路径
filenames=os.listdir(filepath)
for filename in os.listdir(filepath):
    GIF.append(imageio.imread(filepath+"\\"+filename))
imageio.mimsave(filepath+"\\"+'result.gif',GIF,duration=0.3)#这个duration是播放速度，数值越小，速度越快
