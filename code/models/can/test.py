import h5py
import PIL.Image as Image
import numpy as np
import os
import glob
import scipy
from image import *
from model import CANNet
import torch
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error,mean_absolute_error
from torchvision import transforms
import argparse
import json
import matplotlib
matplotlib.use('Agg');

import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib import cm as c


import shutil


transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ])

# the folder contains all the test images

parser = argparse.ArgumentParser(description='PyTorch CANNet')

parser.add_argument('test_json', metavar='test',
                    help='path to val json')

parser.add_argument('output', metavar='VAL',
                    help='path output')

args = parser.parse_args()

def save_dictionary(dictpath_json, dictionary_data):
    a_file = open(dictpath_json, "w")
    json.dump(dictionary_data, a_file, indent=4)
    a_file.close()


vis_path = os.path.join(args.output,'visual_results');
if (os.path.isdir(vis_path)): shutil.rmtree(vis_path, True);
if (os.path.exists(vis_path)):
    raise RuntimeError(f"Output path [{vis_path}] exists but is NOT directory, fatal error");
os.mkdir(vis_path);


checkpoint = torch.load(os.path.join(args.output,'model_best.pth.tar'))
model = CANNet()
model = model.cuda()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

pred= []
gt = []

dictionary_counts={}


# Testing images
img_paths = [];
with open(args.test_json, 'r') as outfile:
    img_paths = json.load(outfile)

# Look-up table for classes
class_lut=[];
with open( os.path.join(os.path.dirname(args.test_json), "dataset.json") , 'r') as outfile:
    class_categories = json.load(outfile)["categories"];

    # Make a sequential list of 
    for i, cat in enumerate(class_categories):
        # Don't implement sorting by id
        if (i != cat["id"]): raise RuntimeError("Categories aren't sorted by id in the dataset JSON. Sort them above this exception or when you make the dataset (or comment this line out)");
    
        # We "know" the classes in the category list are in the order used to generate the ground-truth images for model. Add them to the LUT in that order
        class_lut.append(
            # Also make the categories make me happy
            cat["name"].replace("/", "-").replace(" ", "_").lower()
        );

classes = h5py.File(os.path.join("all", img_paths[0]) + ".gt.h5")['density'].shape[2];


print(f"Detected {classes}:");
for i, cname in enumerate(class_lut):
    print(f"  {i:02d}: {cname}");


# Value of output per class (sum of all)
metric_class_val_out = [ 0 for i in class_lut ];
metric_class_val_gt  = [ 0 for i in class_lut ];
# Every instance of a count, per class
metric_class_out     = [ [] for i in class_lut ];
metric_class_gt      = [ [] for i in class_lut ];
# Every instance of a prediction (over all classes)
metric_img_out       = [];
metric_img_gt        = [];

for img_path in img_paths:
    plain_file=os.path.basename(img_path);
    img = Image.open(os.path.join("all", img_path)).convert('RGB');
    # Half (and floor) image size as that's what the data loader does
    img = img.resize( (int(img.size[0]/2), int(img.size[1]/2)), Image.BICUBIC );
    img = transform(img).cuda();
    img = img.unsqueeze(0);
    
    entire_img=Variable(img.cuda());
    entire_den=model(entire_img).detach().cpu();
    # Stored as [batch, class, x, y]
    den=np.asarray(entire_den[0]);

    
    groundtruth = h5py.File(os.path.join("all", img_path) + ".gt.h5")['density'];
    
    
    # Remove the extension and store it
    plain_file, plain_file_ext = os.path.splitext(plain_file);

    # Save individually the channels of GT and output
    for i, cname in enumerate(class_lut):
        plt.figure(figsize=(16,9), dpi=150);

        count_o  = np.sum(np.sum(den[i]));
        count_gt = np.sum(np.sum(groundtruth[:, :, i]));

        metric_class_val_out[i] += count_o;
        metric_class_val_gt[i]  += count_gt;

        ax = plt.subplot(1,2,1);
        ax.set_title(f"output count={count_o}");
        visden = ax.imshow(den[i], cmap=c.plasma);
        ax.get_figure().colorbar(visden, ax=ax, location="bottom");

        ax = plt.subplot(1,2,2);
        ax.set_title(f"gt count={count_gt}");
        visden = ax.imshow(groundtruth[:, :, i], cmap=c.plasma);
        ax.get_figure().colorbar(visden, ax=ax, location="bottom");
        
        plt.savefig(os.path.join(vis_path, plain_file)+f"_out_{i}_{cname}.png", pad_inches=0.5, bbox_inches='tight');
    
    # Save model input
    plt.imshow(mpimg.imread( os.path.join("boxed_imgs", img_path + ".boxed.jpg")))
    plt.savefig(os.path.join(vis_path, plain_file + '_input' + plain_file_ext),bbox_inches='tight', pad_inches = 0.5, dpi=150)

    #######################################################################
    
    pred_sum = den.sum();#density_1.sum()+density_2.sum()+density_3.sum()+density_4.sum()
    dictionary_counts[plain_file]={ "pred": float(pred_sum), "gt": float(np.sum(groundtruth)) };#round(abs(pred_sum),3)

    print (f"For image {plain_file}");
    for i in range(3):
        print(f"  c_{i}: predicted: {den[i, :, :].sum()}, gt: {np.sum(groundtruth[:, :, i])}\n");

    print(f"  c_ALL: predicted: {pred_sum}, gt: {np.sum(groundtruth)}");

    pred.append(pred_sum)

    gt.append(np.sum(groundtruth))


mae = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

save_dictionary(os.path.join(args.output,"dic_restults.json"),dictionary_counts)

print('MAE: ',mae)
print('RMSE: ',rmse)
results=np.array([mae,rmse])
np.savetxt(os.path.join(args.output,"restults.txt"),results,delimiter=',')
