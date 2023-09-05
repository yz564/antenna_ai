import os
import argparse
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import torch

from typing import NamedTuple
import sys
np.set_printoptions(threshold=sys.maxsize)

from antenna_parameters import PATCH_SIZES, SIXPATCH_TARGETS, SIXPATCH_FREQS, six_patch_antenna

# 1mm -> RESOLUTION pixels
RESOLUTION = 10
WIDTH = 22 * RESOLUTION
HEIGHT = 5 * RESOLUTION
NUM_NODES = 6
STATS_PER_NODE = 2
TARGETS=SIXPATCH_TARGETS
freqs=SIXPATCH_FREQS
#TARGETS = [(2.4, 2.5, -6.0, 1.0), (5.1, 7.0, -6.0, 1.0)]

plt.rc('axes', labelsize=48)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=48)    # fontsize of the tick labels
plt.rc('ytick', labelsize=48)    # fontsize of the tick labels

def convert_to_im(coords: np.ndarray, for_plot: bool=False) -> np.ndarray:
    """assumption is bottom left is 0,0"""

    image = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    # coord is bottom left corner, boundary is size
    coords = np.reshape(coords, (NUM_NODES, STATS_PER_NODE)) * RESOLUTION
    for coord, boundary in zip(coords, PATCH_SIZES):
        # bottom left coord, shouldnt need to be clipped above
        x_bl = int(np.rint(np.clip(coord[0], a_min=0, a_max=WIDTH)))
        y_bl = int(np.rint(np.clip(coord[1], a_min=0, a_max=HEIGHT))) # Flip y-axis

        # top right
        x_tr = int(np.rint(np.clip(coord[0] + RESOLUTION * boundary.width, a_min=0, a_max=WIDTH)))
        y_tr = int(np.rint(np.clip(coord[1] + RESOLUTION * boundary.height, a_min=0, a_max=HEIGHT)))
        for i in range(y_bl, y_tr):
            for j in range(x_bl, x_tr):
                image[HEIGHT - 1 - i][j] = 1.0
                if for_plot: #plot the boundary, only for plot
                    if i == y_bl or i==y_tr-1 or j==x_bl or j==x_tr-1:
                        image[HEIGHT - 1 - i][j] = 0.9
        
    coord, boundary = next(zip(coords, PATCH_SIZES))
    x_bl = int(np.rint(np.clip(coord[0], a_min=0, a_max=WIDTH)))
    y_bl = int(np.rint(np.clip(coord[1], a_min=0, a_max=HEIGHT))) # Flip y-axis
    x_tr = int(np.rint(np.clip(coord[0] + RESOLUTION * boundary.width, a_min=0, a_max=WIDTH)))
    y_tr = int(np.rint(np.clip(coord[1] + RESOLUTION * boundary.height, a_min=0, a_max=HEIGHT)))
    
    for i in range(0, y_bl):
        for j in range(x_bl, x_tr):
            image[HEIGHT - 1 - i][j] = 0.5 # mark the excitation port
    return np.array([image])

def convert_batch_to_im(batch_coords: np.ndarray) -> np.ndarray:
    out = []
    for coords in batch_coords:
        im = convert_to_im(coords)
        out.append(im)
    return np.array(out)

def plot_antenna(im: np.array, pred:np.array, file_name='case.png'):
    fig, ax = plt.subplots(2)
    ax[0].imshow(im.squeeze(axis=0))
    ax[1].set(ylim=(-10,1))
    ax[1].plot(freqs, 20 * np.log10(pred), color='g', marker='o')
    score=six_patch_antenna.calculate_score(pred.reshape(1,-1))
    for fl, fh, t, w in TARGETS:
        ax[1].plot([fl, fh], [t, t], color='b', marker='x')
    plt.title(f'the score is {score}')
    plt.savefig(file_name)
    plt.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To make sure the dataset is collected correctly')
    parser.add_argument('--dataset', '-f', help='training dataset file', default="training_2500.csv")
    parser.add_argument('--im_data_file', '-i', help='file for converted ims', default="training_ims.npy")
    parser.add_argument('--for_plot', '-p', help='plot strips data', default=False, action='store_true')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))

    data_file = os.path.join(base_dir, "data", args.dataset)
    dataset = np.loadtxt(data_file, delimiter=',', dtype=float)
    params = dataset[1:, :12]
    preds = dataset[1:, 12:713:10]
    
    if args.for_plot: 
        for i, x in enumerate(params):
            im = convert_to_im(x, True)
            pred = preds[i]
            plot_antenna(im,pred,file_name=f'image/case_{i}.png')
    else:
        out = convert_batch_to_im(params)
        im_data_file = os.path.join(base_dir, "data", args.im_data_file)
        np.save(im_data_file, out)
    
    '''
    # select the data id to plot its image and s11
    freqs = np.linspace(0,7,71)
    #i=331 #python3 query_dataset.py -f "training_2500.csv" -n 5
    i=2173 
    x=params[i,:]
    fig, ax = plt.subplots(2, 1, figsize=(19, 14))
    im = convert_to_im(x, True)
    pred = preds[i]
    ax[0].imshow(im.squeeze(axis=0))
    #plt.setp(ax[0],xlabel='x [0.1 mm]', fontsize=48)
    #plt.setp(ax[0],ylabel='y [0.1 mm]', fontsize=48)
    ax[0].set_xlabel('x [0.1 mm]', fontsize=48)  # Adjust fontsize as needed
    ax[0].set_ylabel('y [0.1 mm]', fontsize=48)
    ax[1].set(ylim=(-15,1))
    ax[1].plot(freqs, 20 * np.log10(pred), color='g', marker='o',label=f'Simulation', linewidth=6,markersize=12)
    ii=0
    for fl, fh, t, w in TARGETS:
        if ii==0:
            ax[1].plot([fl, fh], [t, t], color='b', marker='o',label=f'Target',linewidth=6,markersize=12, markerfacecolor='none')
            ii+=1
        if ii==1:
            ax[1].plot([fl, fh], [t, t], color='b', marker='o',linewidth=6,markersize=12, markerfacecolor='none')
    #plt.setp(ax[1],xlabel='Frequency [GHz]')
    #plt.setp(ax[1],ylabel='S11 [dB]')
    ax[1].set_xlabel('Frequency [GHz]', fontsize=48)  # Adjust fontsize as needed
    ax[1].set_ylabel('S11 [dB]', fontsize=48) 
    plt.legend(loc=3, fontsize=36, frameon=False)
    #plt.show()
    #plt.savefig(f'example2_best_case_{i}.png')
    plt.savefig(f'example2_best_case_{i}.eps',format='eps')
    plt.close()
    '''
