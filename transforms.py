from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch

class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

### MREF ###
class FrequencyRandomErasing(object):
    def __init__(self, probability=0.5):
        self.probability = probability

    def __call__(self, x):
        if random.uniform(0, 1) > self.probability:
            return x

        ####nxn
        n = 1                   # change here according to the conditions

        a = [int(i * 32 / n) for i in range(n)]
        b= a
        c = a[1:]
        d = a[1:]
        c.append(32)
        d.append(32)

        image_list = []
        for i in range(n):
            for j in range(n):
                imgcrop = x.crop((a[i], b[j], c[i], d[j]))
                image_list.append(imgcrop)
        ####

        result_image_list = []
        ##
        for img in image_list:
            img = np.array(img).astype(np.uint8)
            fft_1 = np.fft.fftshift(np.fft.fftn(img))

            # 랜덤 영역 뽑기
            x_min = np.random.randint(0, 16//n)        # 16
            x_max = np.random.randint(x_min+1, 32//n)        # 32
            y_min = np.random.randint(0, 16//n)        # 16
            y_max = np.random.randint(y_min+1, 32//n)        # 32
            # RE
            fft_1[x_min:x_max, y_min:y_max] = 0

            img = np.fft.ifftn(np.fft.ifftshift(fft_1))
            result_image_list.append(img)

        ####nxn
        row = []
        for ii in range(n):
            r = np.concatenate(result_image_list[ii * n : (ii + 1) * n], axis=0)
            row.append(r)
        new_image = np.concatenate(row, axis=1)
        ####
        x = np.clip(new_image, 0, 255).astype(np.uint8)
        x = Image.fromarray(x)
        # x.show()
        return x