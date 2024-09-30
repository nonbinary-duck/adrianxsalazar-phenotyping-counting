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
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from matplotlib import cm as c
import matplotlib
matplotlib.use('Agg')
plt.style.use('classic')

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

with open(args.test_json, 'r') as outfile:
    img_paths = json.load(outfile)

model = CANNet()

model = model.cuda()



vis_path = os.path.join(args.output,'visual_results');

if (os.path.isdir(vis_path)): shutil.rmtree(vis_path, True);

if (os.path.exists(vis_path)):
    raise RuntimeError(f"Output path [{vis_path}] exists but is NOT directory, fatal error");
os.mkdir(vis_path);


checkpoint = torch.load(os.path.join(args.output,'model_best.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()

pred= []
gt = []

dictionary_counts={}

for img_path in img_paths:
    plain_file=os.path.basename(img_path)
    img = Image.open(os.path.join("all", img_path)).convert('RGB');
    # Half (and floor) image size to save VRAM
    img = img.resize( (int(img.size[0]/2), int(img.size[1]/2)), Image.BICUBIC );
    img = transform(img).cuda();
    img = img.unsqueeze(0)
    
    entire_img=Variable(img.cuda())
    entire_den=model(entire_img)
    entire_den=entire_den.detach().cpu();
    
    pure_name = os.path.splitext(os.path.basename(img_path))[0]
    gt_file = h5py.File(os.path.join("all", img_path) + ".gt.h5")
    groundtruth = np.asarray(gt_file['density'])
    
    den=np.asarray(entire_den.reshape(entire_den.shape[1], entire_den.shape[2], entire_den.shape[3]))

    
    # Remove the extension and store it
    plain_file, plain_file_ext = os.path.splitext(plain_file);
    
    # Clip to range 0-inf.
    visden = np.add(den, np.abs(np.min(den)) );
    # Clip to range 0-1
    visden = np.multiply(visden, 1/np.max(visden));
    # Swap from cxy to xyc
    visden = np.transpose(visden, (1, 2, 0));
    # Plot it
    
    plt.imshow(visden);
    # Save it
    plt.gca().set_axis_off()
    plt.axis('off')
    plt.margins(0,0)
    plt.savefig(os.path.join(vis_path, plain_file)+"_output.png",bbox_inches='tight', pad_inches = 0,dpi=300)
    plt.close()

    

    # Save individually the channels
    for i in range(3):
        plt.imshow(den[i], cmap=c.plasma);
        plt.gca().set_axis_off()
        plt.axis('off')
        plt.margins(0,0)
        plt.savefig(os.path.join(vis_path, plain_file)+f"_output_{i}.png",bbox_inches='tight', pad_inches = 0,dpi=300)
        plt.close()

        # Do colour (isolate the channel so you can see the composite components)
        visden_temp = np.zeros(visden.shape);
        visden_temp[:, :, i] = visden[:, :, i];

        plt.imshow(visden_temp);
        plt.gca().set_axis_off()
        plt.axis('off')
        plt.margins(0,0)
        plt.savefig(os.path.join(vis_path, plain_file)+f"_output_{['r','g','b'][i]}.png",bbox_inches='tight', pad_inches = 0,dpi=300)
        plt.close()
    


    plt.imshow(np.multiply(groundtruth, 1/np.max(groundtruth)));
    plt.gca().set_axis_off()
    plt.axis('off')
    plt.margins(0,0)
    plt.savefig(os.path.join(vis_path, plain_file + "_gt")+".png",bbox_inches='tight', pad_inches = 0,dpi=300)
    plt.close()
    
    # This code needs to be replaced by automatically rendering and linking the boxed images into the dataset
    plt.imshow(mpimg.imread( os.path.join("boxed_imgs", img_path + ".boxed.jpg")))
    plt.gca().set_axis_off()
    plt.axis('off')
    plt.margins(0,0)
    plt.savefig(os.path.join(vis_path, plain_file + '_input' + plain_file_ext),bbox_inches='tight', pad_inches = 0, dpi=300)
    plt.close()

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
