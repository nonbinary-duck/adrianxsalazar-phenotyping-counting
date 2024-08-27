# importing libraries
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
# from scipy.ndimage.filters import gaussian_filter

import scipy
import json
from matplotlib import cm as CM
#from image import *
#from model import CSRNet
import torch
from tqdm import tqdm
import numpy as np
import argparse

import scipy.spatial
from scipy.ndimage import gaussian_filter


class create_density_dataset():

    def __init__(self, dataset_path, beta=0.1):
        self.dataset_path=dataset_path
        self.beta=beta

    def gaussian_filter_density(self, gt):
        """
        Apply a gaussian filter to a density map (gt)
        """
        
        density = np.zeros(gt.shape, dtype=np.float32)
        gt_count = np.count_nonzero(gt)
        if gt_count == 0:
            return density

        pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
        leafsize = 2048
        # build kdtree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=4)

        print ('generate density...')
        for i, pt in enumerate(pts):
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1],pt[0]] = 1.

            #TODO modify this to include more neighbours also the average distance
            if gt_count > 2:
                sigma = (distances[i][1]+distances[i][2]+distances[i][3])*self.beta
            else:
                sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point

            density += gaussian_filter(pt2d, sigma, mode='constant')

        print(f"original count: {np.sum(np.sum(gt))}, new: {np.sum(np.sum(density))}");
        print ('done.')
        return density


    def create_map(self, txt_format=True):
        """
        Create density maps for a dataset
        """

        json_file = open(self.dataset_path, "r");
        dataset = json.load(json_file);
        json_file.close();

        # path=self.dataset_path
        # for img_path in glob.glob(os.path.join(path, '*'+image_format)):
        #     img_paths.append(img_path)

        for image in dataset["images"]:
            img_path = os.path.join("all", image["file_name"]);

            img= plt.imread(img_path)
            k = np.zeros((img.shape[0],img.shape[1]))
            gt=np.loadtxt(img_path + ".txt");

            #The format of the gorund truth is an array with the super pixels e.g.
            #[[x1,y1][x2,y2]]
            # if txt_format== False:
            #     gt = mat["image_info"][0,0][0,0][0]

            for i in range(0,len(gt)):
                if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
                    k[int(gt[i][1]),int(gt[i][0])]=1

            print(img_path)
            k = self.gaussian_filter_density(k)

            x=img_path + ".gt.h5";
            with h5py.File(x, 'w') as hf:
                hf['density'] = k;


    def visualise_density_map(self,path_image):
        """
        Show density plot with matplotlib
        """
        plt.imshow(Image.open(path_image))
        plt.show()
        gt_file = h5py.File(path_image.replace('.png','.h5'),'r')
        groundtruth = np.asarray(gt_file['density'])
        plt.imshow(groundtruth,cmap=CM.jet)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'introduce dataset folder')

    parser.add_argument('-i', metavar='dataset_directory', required=True,

    help='the path to the directory containing the Json file')

    parser.add_argument('-b', metavar='beta or the gaussian filter', required=False,
    help='v')

    args = parser.parse_args()

    # if len(args.b) > 1:
    #     density_map=create_density_dataset(args.i, beta=args.b)
    # else:
    density_map=create_density_dataset(args.i)
    density_map.create_map()
