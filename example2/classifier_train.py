import os
import csv
import random
import time
from typing import Any, List, Tuple
import argparse
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from antenna_parameters import six_patch_antenna, AntennaParameters
from antenna_dataset import SimulationDataSet
from classifier_model import LinearModel, ConvModel

EPSILON = 1E-10
def weighted_mse_loss(pred, truth, weight=1):
    log_truth = 20*torch.log10(truth)
    pred_truth = 20*torch.log10(pred)
    #return torch.mean((pred - log_truth) ** 2)
    return torch.mean((weight * (pred_truth - log_truth)) ** 2)
    
def compute_ce_loss(pred, val, thresh):
    pred=pred.squeeze()
    loss = F.mse_loss(pred, val)
    c1 = (val < thresh)
    c2 = (val >= thresh)
    pos_pred = (pred < thresh)
    neg_pred = (pred >= thresh)

    # c1 = (val >= thresh).unsqueeze(1)
    # c2 = (val < thresh).unsqueeze(1)
    # #truth = torch.concat([c1, c2], dim=1).float()
   
    # gamma = 4
    # pos_loss = c1 * torch.log(pred + EPSILON)
    # neg_loss = c2 * torch.log(1 - pred + EPSILON)
    # focal_loss = torch.mean(-(c1 * (1 - pred) ** gamma * pos_loss + c2 * pred ** gamma * neg_loss))
    # #focal_loss = torch.mean(-(c1 * pos_loss + c2 * neg_loss))

    # pos_pred = (pred > .5).float()
    # neg_pred = (pred < .5).float()
    stats = {}
    if torch.sum(c1) ==0:
        stats["TP"] = 0
    else:
        stats["TP"] = torch.sum(c1 * pos_pred ) / torch.sum(c1) 
    if torch.sum(c2) ==0:
        stats["FP"] = 0
    else:
        stats["FP"] = torch.sum(c2 * pos_pred ) / torch.sum(c2) 
    stats["thresh"]=thresh

    return loss, stats

def train_simulation_model(model, dataset: SimulationDataSet, test_dataset: SimulationDataSet, epochs: int, thresh: float, learningrate:float, classify: bool,
                           device: str, outfile, model_file) -> nn.Module:
    print(f"training: {epochs}, {device}")
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device(device)
    #freq_weight=0.0*np.ones(71)
    freq_weight = np.ones(71)
    freq_weight[23:26]=10
    freq_weight[50:70]=10
    freq_weight=torch.tensor(freq_weight).to(device=device, dtype=torch.float32)

    
    train_size = int(0.9 * len(dataset))
    vali_size = len(dataset) - train_size
    train_dataset, vali_dataset = torch.utils.data.dataset.random_split(dataset, [train_size, vali_size])
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=50, shuffle=True)
    vali_data_loader = DataLoader(dataset=vali_dataset, batch_size=50, shuffle=False)
    test_data_loader = DataLoader(dataset=test_dataset, batch_size=50, shuffle=False)
        #model.to(device=device, dtype=torch.float64)
    model.to(device=device)#, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningrate, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-8, verbose=True)
    train_loss_hist=[]
    train_tp_hist=[]
    train_fp_hist=[]
    vali_loss_hist=[]
    vali_tp_hist=[]
    vali_fp_hist=[]
    best_so_far=1e10
    max_epochs_without_improvement = 50  # Define the maximum number of epochs without improvement
    current_epochs_without_improvement = 0  # Initialize the counter
    for epoch in range(epochs):
        model.train()
        train_loss = torch.tensor(0.0, device=device)
        train_tp = torch.tensor(0.0, device=device)
        train_fp = torch.tensor(0.0, device=device)
        train_steps = 0
        for x, v, p, w in train_data_loader:
            pred, val = model(x.to(device=device, dtype=torch.float32))

            #freq_weight = w.to(device=device, dtype=torch.float32)
            #loss=0
            
            if classify:
                ce_loss, stats = compute_ce_loss(val,  v.to(device=device, dtype=torch.float32), thresh)
                loss = ce_loss
            else:
                loss = weighted_mse_loss(pred, p.to(device=device, dtype=torch.float32), weight=freq_weight)
                val = torch.tensor(six_patch_antenna.calculate_score(pred.detach().numpy()))
                ce_loss, stats = compute_ce_loss(val.to(device=device, dtype=torch.float32),  v.to(device=device, dtype=torch.float32), thresh)
            train_tp += stats["TP"]
            train_fp += stats["FP"]
            train_loss += loss
            train_steps += 1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print (pred[0])
        #print (p[0])
        #print (val)
        #print (v)
        model.eval()
        with torch.no_grad():
            vali_loss = torch.tensor(0.0, device=device)
            vali_tp = torch.tensor(0.0, device=device)
            vali_fp = torch.tensor(0.0, device=device)
            vali_steps = 0
            for x, v, p, w in vali_data_loader:
                pred, val = model(x.to(device=device, dtype=torch.float32))
                #loss = F.mse_loss(pred, p.to(device=device, dtype=torch.float64))
                #weight = 1/v.clip(0.00001)
                #freq_weight = w.to(device=device, dtype=torch.float32)

                if classify:
                    ce_loss, stats = compute_ce_loss(val,  v.to(device=device, dtype=torch.float32), thresh)
                    loss = ce_loss
                else:
                    loss = weighted_mse_loss(pred, p.to(device=device, dtype=torch.float32), weight=freq_weight)
                    val = torch.tensor(six_patch_antenna.calculate_score(pred.detach().numpy()))
                    ce_loss, stats = compute_ce_loss(val.to(device=device, dtype=torch.float32),  v.to(device=device, dtype=torch.float32), thresh)
                #loss = ce_loss + mse_loss
                vali_tp += stats["TP"]
                vali_fp += stats["FP"]
                vali_loss += loss
                vali_steps += 1

        
        print(f"epoch {epoch} loss: {train_loss.item() / train_steps}, vali loss: {vali_loss.item() / vali_steps}")
        print(f"epoch {epoch} train true positive: {train_tp.item() / train_steps}, vali true positive: {vali_tp.item() / vali_steps}")
        print(f"epoch {epoch} train false positive: {train_fp.item() / train_steps}, vali false positive: {vali_fp.item() / vali_steps}")
        train_tp_hist.append(train_tp.item() / train_steps)
        vali_tp_hist.append(vali_tp.item() / vali_steps)
        train_fp_hist.append(train_fp.item() / train_steps)
        vali_fp_hist.append(vali_fp.item() / vali_steps)
        outfile.writerow([train_loss.item() / train_steps, vali_loss.item() / vali_steps, train_tp.item() / train_steps, vali_tp.item() / vali_steps,train_fp.item() / train_steps, vali_fp.item() / vali_steps])
        train_loss_hist.append(train_loss.item() / train_steps)
        vali_loss_hist.append(vali_loss.item() / vali_steps)
        
        if vali_loss.item()<best_so_far:
            best_so_far=vali_loss.item()
            torch.save(model, f"{model_file}")
            print(f"model saved to {model_file}")
            current_epochs_without_improvement = 0
        else:
            current_epochs_without_improvement += 1
        if current_epochs_without_improvement >= max_epochs_without_improvement:
            print("Early stopping triggered.")
            epochs=epoch+1
            break
        scheduler.step(vali_loss)

    model.eval()
    test_loss = 0
    test_steps = 0
    test_tp = torch.tensor(0.0, device=device)
    test_fp = torch.tensor(0.0, device=device)
    mse_te_loss = torch.tensor(0.0, device=device)
    ce_te_loss = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for x, v, p, w in test_data_loader:
            pred, val = model(x.to(device=device, dtype=torch.float32))
            #freq_weight = w.to(device=device, dtype=torch.float32)
            if classify:
                ce_loss, stats = compute_ce_loss(val,  v.to(device=device, dtype=torch.float32), thresh)
                loss = ce_loss
            else:
                loss = weighted_mse_loss(pred, p.to(device=device, dtype=torch.float32), weight=freq_weight)
                val = torch.tensor(six_patch_antenna.calculate_score(pred.detach().numpy()))
                ce_loss, stats = compute_ce_loss(val.to(device=device, dtype=torch.float32),  v.to(device=device, dtype=torch.float32), thresh)
            test_tp += stats["TP"]
            test_fp += stats["FP"]
            test_loss += loss
            test_steps += 1
    test_loss /= test_steps
    print(f"test loss: {test_loss}")
    print(f"test true positive rate: {test_tp.item() / test_steps}")
    print(f"test false positive rate: {test_fp.item() / test_steps}")
    #outfile.writerow([test_loss, 0])
    fig, ax = plt.subplots(3, 1)
    ax[0].semilogy(np.arange(epochs),train_loss_hist, color='r',label='train loss')
    ax[0].semilogy(np.arange(epochs),vali_loss_hist, color='b',label='validation loss')

    ax[1].plot(np.arange(epochs),train_tp_hist, color='r',label='train true positive rate')
    ax[1].plot(np.arange(epochs),vali_tp_hist, color='b',label='validation true positive rate')

    ax[2].plot(np.arange(epochs),train_fp_hist, color='r',label='train false positive rate')
    ax[2].plot(np.arange(epochs),vali_fp_hist, color='b',label='validation false positive rate')

    ax[0].legend()
    ax[1].legend()
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('MSE loss')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel(f'True Positive Rate, viable thresh: {stats["thresh"]}')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel(f'False Positive Rate, viable thresh: {stats["thresh"]}')
    plt.tight_layout()
    plt.show()
    return model

'''
def test_simulation_model(model_file: str, meta_file: str, test_dataset: SimulationDataSet, num_nodes: int, stats_per_node: int):
    with open(meta_file, "rb") as f:
        freqs, targets = pickle.load(f)
    func = AntennaModelFunc(AntennaFunc.GRP2_BOUNDS, num_nodes, stats_per_node, model_file, freqs, targets)
    mse = np.square(func(test_dataset.xs)[1] - test_dataset.preds).mean(axis=1)
    print(f"test dataset MSE: {mse.mean()}, {len(mse)}")
'''

def plot_comparison(model: str, antenna_parameters: AntennaParameters, dataset: SimulationDataSet, outfile: str, device='cpu', num_plot: int=5):
    freqs = antenna_parameters.freqs
    targets = antenna_parameters.targets
    top = sorted(enumerate(dataset.values), key=lambda x:x[1])
    top_ant = []
    j = 0
    for i in range(num_plot):
        ident, val = top[j]
        top_ant.append(ident)
        while top[j][1] == val:  
            j += 1
    choices = top_ant + list(random.sample(range(0, len(dataset)), num_plot))
    fig, ax = plt.subplots(num_plot*2, figsize=(18,24))
    l1 = None
    l2 = None
    for count, i in enumerate(choices):
        simulation_pred = dataset.preds[i]
        model_pred, _ = model(torch.from_numpy(dataset.xs[i:i+1]).to(device=device, dtype=torch.float32))
        model_pred = model_pred.cpu().detach().numpy().flatten()
        #const, zeros, poles = model.get_zeros_poles_c(torch.from_numpy(dataset.xs[i:i+1]).to(device=device, dtype=torch.float32))
        #const = const.cpu().detach().numpy()
        #zeros = zeros.cpu().detach().numpy()
        #poles = poles.cpu().detach().numpy()
        ##if count < num_plot:
        ##    place = 2 * count + 1
        ##else:
        ##    place = 2 * (count % num_plot) + 2
        l1 = ax[count].plot(freqs, 20 * np.log10(model_pred), color='r', marker='x')
        l2 = ax[count].plot(freqs, 20 * np.log10(simulation_pred), color='g', marker='+')
        #l3 = ax[count, 1].plot(freqs, model_pred.flatten(), color='r', marker='x')
        #l4 = ax[count, 1].plot(freqs, simulation_pred, color='g', marker='+')


        #ax[count, 1].scatter(const.real[0], const.imag[0], color='k', marker='s', label='Const')
        #ax[count, 1].scatter(zeros.real[0], zeros.imag[0], color='b', marker='*', label='Zeros')
        #ax[count, 1].scatter(poles.real[0], poles.imag[0], color='y', marker='x', label='Poles')
        #ax[count, 1].grid(visible=True, which='major')
        #l1 = plt.plot(freqs, model_pred.flatten(), color='r', marker='x')
        #l2 = plt.plot(freqs, simulation_pred, color='g', marker='+')
    #kkkax[0,1].legend()
    plt.figlegend((l1[0], l2[0]), ('Prediction', 'Simulation'), 'upper center')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(outfile)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Function MCTS example')
    parser.add_argument('--model_name', '-m', help='base name for saved models', default="antenna_cnn00")
    parser.add_argument('--backbone', '-b', help='bakbone_file', default=None)
    parser.add_argument('--training_dataset', '-f', help='training dataset file', default="data_iters.csv")
    parser.add_argument('--test_dataset', '-t', help='test dataset file', default="test_dataset.csv")
    parser.add_argument('--im_data_file', '-i', help='file for training  ims', default="training_ims.npy")
    parser.add_argument('--test_im_data_file', '-x', help='file for test  ims', default="test_ims.npy")
    parser.add_argument('--epoch', '-e', type=int, help='number of epochs', default=200)
    parser.add_argument('--threshold', '-th', type=float, help='threshhold score to be acceptable', default=5.0)
    parser.add_argument('--learningrate', '-lr', type=float, help='the initial learning rate', default=0.001)
    parser.add_argument('--num-poles', '-n', type=int, help='number of poles', default=2)
    parser.add_argument('--num-zeros', '-z', type=int, help='number of zeros', default=2)
    parser.add_argument('--device', '-d', type=str, help='device', default='cpu')
    parser.add_argument('--plot', '-p', help='plot data', default=False, action='store_true')
    parser.add_argument('--classify', '-c', help='train classifer', default=False, action='store_true')

    args = parser.parse_args()

    num_poles = args.num_poles
    num_zeros = args.num_zeros
    model_name = args.model_name
    device = args.device

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_file = os.path.join(base_dir, "data", args.training_dataset)
    test_data_file = os.path.join(base_dir, "data", args.test_dataset)
    model_file = os.path.join(base_dir, "data", f"{model_name}.pth")
    meta_file = os.path.join(base_dir, "data", f"{model_name}.pkl")
    backbone_file = None
    if args.backbone:
        backbone_file = os.path.join(base_dir, "data", f"{args.backbone}.pth")
    antenna_parameters = six_patch_antenna
    train_dataset = SimulationDataSet(train_data_file, antenna_parameters)
    test_dataset = SimulationDataSet(test_data_file, antenna_parameters)
    im_train_file = os.path.join(base_dir, "data", args.im_data_file)
    im_test_file = os.path.join(base_dir, "data", args.test_im_data_file)
    train_dataset.add_images(im_train_file)#, augment=True)
    test_dataset.add_images(im_test_file)
    #im_outfile = f"{args.model_name}"
    #train_dataset.upsample()
    if args.plot:
        model = torch.load(f"{model_file}", map_location=torch.device('cpu'))
        plot_comparison(model, antenna_parameters, train_dataset, model_name + '_tr.png')
        plot_comparison(model, antenna_parameters, test_dataset, model_name + '_te.png')
    else:
        input_space = train_dataset.xs.shape[1]
        pred_space = train_dataset.preds.shape[1]
   #            #model = FourierFeatureModel(pred_space, hidden_size=1024, B_height=args.b_feat, std=args.std)
        model = ConvModel(pred_space, hidden_size=1024)
   #     else:
   #         model = SimulationLinearModel(input_space, pred_space, hidden_size)

        outfile_p = os.path.join(base_dir, f"{model_name}.csv")
        outfile = open(outfile_p, 'w')
        outfile_writer = csv.writer(outfile)
        print(f'threshold is {args.threshold}')
        train_simulation_model(model, train_dataset, test_dataset, args.epoch, args.threshold, args.learningrate, args.classify, device, outfile_writer,model_file)
        outfile.close()
        #torch.save(model, f"{model_file}")
        #print(f"model saved to {model_file}")

        # plot_comparison(model, antenna_parameters, train_dataset, model_name + '_tr.png', device)
        # #test_simulation_model(model_file, meta_file, test_dataset, num_nodes, stats_per_node)
        # plot_comparison(model, antenna_parameters, test_dataset, model_name + '_te.png', device)

