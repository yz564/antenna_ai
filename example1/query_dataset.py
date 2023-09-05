import os
import argparse
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from antenna_dataset import SimulationDataSet
import torch
from antenna_parameters import PATCH_SIZES, FIVEPATCH_TARGETS, five_patch_antenna
from convert_to_im import convert_batch_to_im, plot_antenna, convert_to_im

SMALL_SIZE = 48
MEDIUM_SIZE = 48
BIGGER_SIZE = 48

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def check_plot_comparison(dataset: SimulationDataSet, case_id: int):
    targets = five_patch_antenna.targets
    freqs = five_patch_antenna.freqs
    func = AntennaSimulatorFunc(five_patch_antenna)
    # choices = random.sample(range(0, len(dataset)), 6)
    print(f"Checking data # : {case_id}")
    choices=np.array([case_id])
    plt.figure(figsize=(12, 9))
    count = 1
    l1 = None
    l2 = None
    for i in choices:
        dataset_pred = dataset.preds[i]
        #_, model_pred = func(np.array(dataset.xs[i]).reshape((1, -1)))
        # plt.subplot(3, 2, count)
        l1 = plt.plot(freqs, 20 * np.log10(dataset_pred), color='r', marker='+')
        #l2 = plt.plot(freqs, 20 * np.log10(model_pred.flatten()), color='g', marker='x')
        for fl, fh, t, w in targets:
            l3=plt.plot([fl, fh], [t, t], color='k', marker='o')
        count += 1
    #plt.figlegend((l2[0],l3[0]), ('Simulation again to check','Target'), 'upper center')
    #plt.figlegend(l3[0], 'Target', 'upper center')
    plt.legend()
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(f"plot_comparison{case_id}")
    plt.close()
    
def plot_scores(train_dataset: SimulationDataSet, model_name=None, classify = True):
    goal_values=train_dataset.values
    ids=np.argsort(goal_values)
    simulated_scores=goal_values[ids]

    plt.figure(figsize=(12, 9))
    plt.plot(range(len(ids)),simulated_scores,color='r',label='simulation')
    if model_name:
        classifier_file = os.path.join(base_dir, "data", f"{args.model_name}.pth")
        classifier = torch.load(classifier_file, map_location=torch.device('cpu'))
        xs=train_dataset.xs
        ims = convert_batch_to_im(xs)
        pred, viable = classifier(torch.from_numpy(ims).to(dtype=torch.float32))
        if (not classify):
            viable=torch.tensor(five_patch_antenna.calculate_score(pred.detach().numpy()))
        plt.plot(range(len(ids)),viable.detach().numpy()[ids],color='b',label='CNN prediction')
    plt.axhline(y=3.93, color='r', linestyle='--')
    plt.ylim((0,6))
    plt.xlabel('sorted case id')
    plt.ylabel('score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Scores_{model_name}")
    plt.close()

def plot_scores_all(train_dataset: SimulationDataSet):
    casenum=500
    goal_values=train_dataset.values
    N=int(len(goal_values)/casenum)
    fig,ax=plt.subplots(figsize=(12, 9))
    medians=[]
    for i in range(N):
        plt.plot((i)*np.ones(casenum), np.array(goal_values[0+casenum*i:casenum+casenum*i]),'+',markersize=7)
        medians.append(np.median(goal_values[0+casenum*i:casenum+casenum*i]))
    plt.plot(range(N),medians,color='k',marker='o',label='median')
    for index in range(N):
        ax.text(index, medians[index], "{:.2f}".format(medians[index]), size=16)
    plt.ylim((0,6))
    plt.xlabel('iteration id')
    plt.ylabel('score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Scores_All")
    plt.close()


    fig,ax=plt.subplots(figsize=(12, 9))
    ax.hist(goal_values,bins=60)
    plt.xlim((0,6))
    
    plt.xlabel('score')
    plt.ylabel('antenna amount')
    plt.tight_layout()
    plt.savefig(f"Histogram_All")
    plt.close()

def plot_scores_all_select_positions(train_dataset: SimulationDataSet):
    #func = AntennaSimulatorFunc(AntennaFunc.GRP2_BOUNDS, freqs, targets)
    casenum=500
    goal_values=train_dataset.values
    N=int(len(goal_values)/casenum)
    fig,ax=plt.subplots(figsize=(12, 9))
    medians=[]
    acc_medians=[6]
    for i in range(N):
        plt.plot((i+1)*np.ones(casenum), np.array(goal_values[0+casenum*i:casenum+casenum*i]),'+',markersize=16)
        medians.append(np.median(goal_values[0+casenum*i:casenum+casenum*i]))
        acc_medians.append(np.median(goal_values[0:casenum+casenum*i]))
    plt.plot(range(1,N+1),medians,color='k',marker='o', markersize=12,label='Generated data median')
    plt.plot(range(1,N+1),acc_medians[:N],color='r',marker='^',markersize=12,label='Classifier threshold')
    for index in range(N):
        ax.text(index+1, medians[index], "{:.1f}".format(medians[index]), size=38, ha='center', va='top')
        ax.text(index+1, acc_medians[index],"{:.1f}".format(acc_medians[index]), size=38, color='r', ha='center', va='bottom')
    plt.ylim((0,6))
    plt.xticks(range(1,N+1),["1","2","3","4","5","6","7","8"])
    plt.xlabel('Classifier iteration [k]')
    plt.ylabel('Score')
    plt.legend(loc=3, fontsize=32, frameon=False)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"example1_select_positions_points")
    plt.savefig(f"example1_select_positions_points.eps",format='eps')
    plt.close()

    plt.figure(figsize=(12, 9))
    fig,ax=plt.subplots(figsize=(12, 9))
    handles = []
    for i in range(N):
        this_set_values=goal_values[0+casenum*i:casenum+casenum*i]
        ids=np.argsort(this_set_values)
        simulated_scores=this_set_values[ids]
        line, =plt.plot(range(len(ids)),simulated_scores,label=f'Iter {i+1}',linewidth=6)
        handles.append(line)
    plt.scatter([250,250,250,250,250,250,250,250],medians,color='k',marker='o', s=50, zorder=2)
    '''
    for index in range(N):
        ax.text(250, medians[index], "{:.2f}".format(medians[index]), size=48)
    '''
    
    plt.ylim((0,6))
    plt.xlabel('Sorted case id [j]')
    plt.ylabel('Score')
    plt.legend(loc=4, fontsize=32,ncol=2, frameon=False)
    plt.tight_layout()
    #plt.grid()
    #plt.show()
    plt.savefig(f"example1_select_positions_curves")
    plt.savefig(f"example1_select_positions_curves.eps",format='eps')
    plt.close()

    fig,ax=plt.subplots(figsize=(12, 9))
    i=0
    ax.hist(goal_values[0+casenum*i:casenum+casenum*i],bins=60)
    plt.xlim((0,6))
    
    plt.xlabel('Score')
    plt.ylabel('Case amount')
    plt.tight_layout()
    #plt.grid()
    #plt.show()
    plt.savefig(f"example1_histogram_initial_500")
    plt.savefig(f"example1_histogram_initial_500.eps",format='eps')
    plt.close()

    fig,ax=plt.subplots(figsize=(12, 9))
    i=N-1
    ax.hist(goal_values[0+casenum*i:casenum+casenum*i],bins=60)
    plt.xlim((0,6))
    
    plt.xlabel('Score')
    plt.ylabel('Case amount')
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"example1_histogram_last_500")
    plt.savefig(f"example1_histogram_last_500.eps",format='eps')
    plt.close()


def plot_scores_all_select_dimensions(train_dataset: SimulationDataSet):
    #func = AntennaSimulatorFunc(AntennaFunc.GRP2_BOUNDS, freqs, targets)
    casenum=100
    goal_values=train_dataset.values
    N=int(len(goal_values)/casenum)
    fig,ax=plt.subplots(figsize=(12, 9))
    medians=[]
    for i in range(N):
        plt.plot((i+1)*np.ones(casenum), np.array(goal_values[0+casenum*i:casenum+casenum*i]),'+', markersize=16)
        medians.append(np.median(goal_values[0+casenum*i:casenum+casenum*i]))
    #plt.scatter(range(1,N+1),medians, s=400,color='k',marker='*', label='Median')
    for index in range(N):
        ax.text(index+1, medians[index], "{:.2f}".format(medians[index]), size=48,  ha='center', va='bottom')
    plt.scatter(range(1,N+1),medians, s=400,color='k', marker='*', zorder=2, label='Median')
    plt.ylim((0,6))
    plt.xticks(range(1,N+1),["1","2","3","4","5"])
    plt.xlabel('Dimension set [i]')
    plt.ylabel('Score')
    plt.legend(loc=4, fontsize=32, frameon=False)
    plt.tight_layout()
    #plt.show()
    plt.savefig(f"example1_select_dimensions_points")
    plt.savefig(f"example1_select_dimensions_points.eps",format='eps')
    plt.close()

    plt.figure(figsize=(12, 9))
    fig,ax=plt.subplots(figsize=(12, 9))
    marker_position = casenum // 2
    for i in range(N):
        this_set_values=goal_values[0+casenum*i:casenum+casenum*i]
        ids=np.argsort(this_set_values)
        simulated_scores=this_set_values[ids]
        plt.plot(range(len(ids)),simulated_scores,label=f'set {i+1}',linewidth=6)
    #plt.scatter([50,50,50,50,50],medians,s=400,color='k',marker='*')
    #for index in range(N):
        #ax.text(50, medians[index], "{:.2f}".format(medians[index]), size=36)
        # Annotate the median value on the curve
        '''
        if index == 4:
            ax.annotate("{:.2f}".format(medians[index]), xy=(marker_position, medians[index]), xytext=(-50, -30), textcoords='offset points', size=48, va='bottom', ha='left')
        else:
            ax.annotate("{:.2f}".format(medians[index]), xy=(marker_position, medians[index]), xytext=(50, -80), textcoords='offset points', size=48, va='bottom', ha='right')
        '''
    plt.ylim((2,6))
    plt.xlabel('Sorted case id [j]')
    plt.ylabel('Score')
    plt.legend(loc=4, fontsize=32, ncol=2, frameon=False)
    plt.tight_layout()
    plt.scatter([50,50,50,50,50],medians,s=400,color='k', marker='*', zorder=2)
    #plt.show()
    plt.savefig(f"example1_select_dimensions_curves")
    plt.savefig(f"example1_select_dimensions_curves.eps",format='eps')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='To make sure the dataset is collected correctly')
    parser.add_argument('--model_name', '-m', help='base name for saved models', default=None)
    parser.add_argument('--training_dataset', '-f', help='training dataset file', default="training_dataset_4000.csv")
    parser.add_argument('--id', '-i',type=int, help='id of data to check', default=0)
    parser.add_argument('--number', '-n',type=int, help='number of data to check', default=0)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_data_file = os.path.join(base_dir, "data", args.training_dataset)

    train_dataset = SimulationDataSet(train_data_file, five_patch_antenna)
    goal_values=train_dataset.values

    min_value = np.min(goal_values)
    min_id=np.argmin(goal_values)
    print(f"The minimum score of the dataset is : {min_value}, the id of this data is {min_id}")
    print(f"The median score of the {len(goal_values)} dataset is : {np.median(goal_values)}")
    ids=np.argsort(goal_values)
    choices=ids[0:args.number]
    for j, i in enumerate(choices):
        print(f"{j+1}: score is {goal_values[i]}, case id is {i}")
        x=train_dataset.xs[i]
        im = convert_to_im(x)
        pred = train_dataset.preds[i]
        plot_antenna(im,pred,file_name=f'best_{j}_case_{i}.png')

    #plot_scores(train_dataset, args.model_name) #python3 query_dataset.py -m "antenna_cnn07_c" -f "iter7_500.csv" 
    #plot_scores_all(train_dataset)

    #plot_scores_all_select_positions(train_dataset)  # python3 query_dataset.py -f training_dataset_4000.csv
    #plot_scores_all_select_dimensions(train_dataset)  # python3 query_dataset.py -f initial.csv

    #if args.id==0:
        #check_plot_comparison(train_dataset, min_id)
    #else:
        #check_plot_comparison(train_dataset, args.id)
