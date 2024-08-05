# importing libraries
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from matplotlib import cm as CM
#from image import *
#from model import CSRNet
import torch
from tqdm import tqdm
import numpy as np
import argparse


def visualise_density_map(path_image):
    """
    Show density plot with matplotlib
    """
    plt.imshow(Image.open(path_image))
    plt.show()
    gt_file = h5py.File(path_image,'r')
    groundtruth = np.asarray(gt_file['density'])
    plt.imshow(groundtruth,cmap=CM.jet)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'introduce dataset folder')

    parser.add_argument('-i', metavar='density', required=True, help='The path to the density file')

    args = parser.parse_args()

    # if len(args.b) > 1:
    #     density_map=create_density_dataset(args.i, beta=args.b)
    # else:
    visualise_density_map(args.i)
