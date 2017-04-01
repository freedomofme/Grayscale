# -*- coding: utf-8 -*-
## @package som_cm.results.som_single_image
#
#  Demo for single image.
#  @author      tody
#  @date        2015/08/31

import os
import numpy as np
import matplotlib.pyplot as plt
from core.color_pixels import ColorPixels
from mpl_toolkits.mplot3d import Axes3D


from io_util.image import loadRGB
from results.resu import batchResults, resultFile
from core.hist_3d import Hist3D
from core.som import SOMParam, SOM, SOMPlot
from som_cm.plot.window import showMaximize


## Setup SOM in 1D and 2D for the target image.
def setupSOM(image, random_seed=100, num_samples=2000):
    np.random.seed(random_seed)

    hist3D = Hist3D(image, num_bins=16)
    color_samples = hist3D.colorCoordinates()

    # 原来代码是这样的: len(color_samples) - 1, 但是我认为不用减1
    random_ids = np.random.randint(len(color_samples), size=num_samples)
    # 同时我觉得这里不用再采样1000了，因为之前已经采样了，保证不会大于1000，这样做反而认为造成数据冗余, 在ColorPixels.py中
    samples = color_samples[random_ids]

    # 测试了以下代码，效果几乎一样
    # samples = color_samples

    # 删除像素点
    # bl=samples==[255,255,255]
    # bl=np.any(bl,axis=1)
    # ind=np.nonzero(bl)[0]
    # samples = np.delete(samples,ind,axis=0)
    # print len(samples)
    #
    # bl=samples==[254,255,255]
    # bl=np.any(bl,axis=1)
    # ind=np.nonzero(bl)[0]
    # samples = np.delete(samples,ind,axis=0)



    #1000
    print len(samples)

    param1D = SOMParam(h=64, dimension=1)
    som1D = SOM(samples, param1D)

    param2D = SOMParam(h=32, dimension=2)
    som2D = SOM(samples, param2D)
    return som1D, som2D


## Demo for the single image file.
def singleImageResult(image_file):
    image_name = os.path.basename(image_file)
    image_name = os.path.splitext(image_name)[0]

    image = loadRGB(image_file)

    som1D, som2D = setupSOM(image)

    fig = plt.figure(figsize=(12, 10))
    fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.9, wspace=0.1, hspace=0.2)

    font_size = 15
    fig.suptitle("SOM-Color Manifolds for Single Image", fontsize=font_size)

    plt.subplot(331)
    h, w = image.shape[:2]
    plt.title("Original Image: %s x %s" % (w, h), fontsize=font_size)
    plt.imshow(image)
    plt.axis('off')

    print "  - Train 1D"
    som1D.trainAll()

    print "  - Train 2D"
    som2D.trainAll()

    som1D_plot = SOMPlot(som1D)
    som2D_plot = SOMPlot(som2D)
    plt.subplot(332)
    plt.title("SOM 1D", fontsize=font_size)
    # 如果改变updateImage函数的返回值，那么可以用以下语句，代替以下第二行语句。
    # plt.imshow(som1D_plot.updateImage())
    som1D_plot.updateImage()
    plt.axis('off')

    plt.subplot(333)
    plt.title("SOM 2D", fontsize=font_size)
    som2D_plot.updateImage()
    plt.axis('off')


    color_pixels = ColorPixels(image)
    pixels = color_pixels.pixels(color_space="rgb")
    ax = fig.add_subplot(334, projection='3d')
    plt.title("cloudPoint", fontsize=font_size)
    som1D_plot.plotCloud(ax, pixels)

    hist3D = Hist3D(image, num_bins=16)
    color_samples = hist3D.colorCoordinates()
    ax = fig.add_subplot(337, projection='3d')
    plt.title("cloudPoint", fontsize=font_size)
    som1D_plot.plotCloud(ax, color_samples)



    ax1D = fig.add_subplot(335, projection='3d')
    plt.title("1D in 3D", fontsize=font_size)
    som1D_plot.plot3D(ax1D)

    ax2D = fig.add_subplot(336, projection='3d')
    plt.title("2D in 3D", fontsize=font_size)
    som2D_plot.plot3D(ax2D)

    plt.subplot(338)
    plt.title("Gray", fontsize=font_size)

    # 如果改变updateImage函数的返回值，那么可以用以下语句，代替以下第二行语句。
    a,b = som2D_plot.showGrayImage2(image)
    plt.imshow(a, cmap='gray', vmin = 0, vmax = 1)
    plt.axis('off')

    plt.subplot(339)
    plt.title("Gray", fontsize=font_size)
    plt.imshow(b, cmap='gray', vmin = 0, vmax = 1)


    result_file = resultFile("%s_single" % image_name)
    plt.savefig(result_file)
    #showMaximize()


## Demo for the given data names, ids.
def singleImageResults(data_names, data_ids):
    batchResults(data_names, data_ids, singleImageResult, "SOM (single image)")

if __name__ == '__main__':
    data_names = ["apple", "banana", "tulip", "sky", "flower"]
    data_ids = [0, 1, 2]

    singleImageResults(data_names, data_ids)