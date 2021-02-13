import os, os.path
import math
from PIL import Image
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import numpy


def dct_2_dim(d2image, default="ortho"):
    return dct(dct(d2image.T, norm=default).T, norm=default)


def i_dct_2_dim(d2image, default="ortho"):
    return idct(idct(d2image.T, norm=default).T, norm=default)


image_list = []
mod_image_list = []

#for directory in os.listdir('Mask/Unmodified/'):
#    for file in os.listdir('Mask/Unmodified/' + directory):
#        image_list.append(numpy.array(Image.open('Mask/Unmodified/'+directory+'/'+file).convert("L")))

for file in os.listdir('Mask/tmp/'):
    image_list.append(numpy.array(Image.open('Mask/tmp/' + file).convert("L")))

for px in image_list:
    image_size = px.shape
    t_px = numpy.zeros(image_size)
    co_thresh = 0.1
    pixel_size = 8

    for i in numpy.r_[:image_size[0]:pixel_size]:
        for j in numpy.r_[:image_size[1]:pixel_size]:
            t_px[i:(i+pixel_size):1, j:(j+pixel_size):1] = dct_2_dim(px[i:(i+pixel_size), j:(j+pixel_size)])

    threshed_t_px = t_px * (abs(t_px) > (co_thresh*numpy.max(t_px)))

    print((numpy.sum(threshed_t_px == 0.0) / float(image_size[0]*image_size[1]))*100)

    mod_px = numpy.zeros(image_size)

    for i in numpy.r_[:image_size[0]:pixel_size]:
        for j in numpy.r_[:image_size[1]:pixel_size]:
            mod_px[i:(i+pixel_size):1, j:(j+pixel_size):1] = i_dct_2_dim(threshed_t_px[i:(i+pixel_size),
                                                                         j:(j+pixel_size)])

    mod_image_list.append(mod_px)

for image_index in range(1, len(mod_image_list)+1):
    picture_name = "%03d" % image_index
    Image.fromarray(mod_image_list[image_index-1]).convert('RGB').save('Mask/data/test/' +
                                                                       picture_name + '/8px01thresh', 'JPEG')
    Image.fromarray(image_list[image_index-1]).convert('RGB').save('Mask/Unmodified/' +
                                                                   picture_name + '/original', 'JPEG')

for image_index in range(len(mod_image_list)):
    plt.subplot(5, math.ceil(len(mod_image_list)/5)+1, image_index+1)
    plt.imshow(image_list[image_index], cmap='gray')

plt.show()
