import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sns
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

from utils._model_results import _get_model_results, _evaluate_model
from utils._noising import _dataset_noising
from utils._gradcam import _save_dataset_gradcam
from utils._resnet import resnet18_8
from utils._preprocessing import _preprocessing_dataset_noising, _get_dataloader



################# Prepare Part ###################

def random_rotation_matrix():
    # Generate a random angle between 0 and 2*pi (360 degrees)
    angle = np.random.rand() * 2 * np.pi

    # Create the 2D rotation matrix
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return rotation_matrix


def apply_rotation(trajectory_data, rotation_matrix):
    # Apply the rotation to the trajectory data, shape (2,1000)
    rotated_trajectory_data = np.dot(rotation_matrix, trajectory_data)
    return rotated_trajectory_data


def _get_rot_aug_dataset(model_wise_traj, model_wise_label, model_wise_exp, model_wise_feat, percentile, aug_size):
    feat = np.mean(model_wise_feat, axis=-1)
    
    # Select the top n% of trajectories based on the feature
    feat_idx = np.argsort(feat)[::-1][:int(percentile * 0.01 * len(feat))]
    
    # Randomly select additional trajectories to have a total of 10,000 datasets
    if percentile==100:
        feat_idx_add = np.random.choice(feat_idx, aug_size*100)
    else:
        feat_idx_add = feat_idx
    
    feat_rotated_traj = [
        np.array([apply_rotation(traj[0], random_rotation_matrix())])
        for traj in model_wise_traj[feat_idx_add]]
    
    feat_aug_rot = []
    feat_aug_rot_label = []
    feat_aug_rot_exp = []
    
    feat_aug_rot.extend(model_wise_traj)
    feat_aug_rot.extend(feat_rotated_traj)
    
    feat_aug_rot_label.extend(model_wise_label)
    feat_aug_rot_label.extend(model_wise_label[feat_idx_add])
    
    
    feat_aug_rot_exp.extend(model_wise_exp)
    feat_aug_rot_exp.extend(model_wise_exp[feat_idx_add])


    return feat_aug_rot, feat_aug_rot_label, feat_aug_rot_exp


def _save_aug_dataset_rot(dataset, gradcam, percentile, aug_size, noise=False):
    if noise==False:
        save_path = f"./dataset_noiseless/aug{aug_size}/0_p{percentile}.npy"
    else:
        save_path = f"./dataset/aug{aug_size}/0_p{percentile}.npy"
        
    if os.path.exists(save_path):
        print("Exist")
        return
        
    traj_dataset = dataset[0]
    label_dataset = dataset[1]
    exp_dataset = dataset[2]

    aug_traj_dataset = []
    aug_label_dataset = []
    aug_exp_dataset = []
    for i in range(8):
        model_wise_traj = traj_dataset[i * 10000:(i + 1) * 10000]
        model_wise_label = label_dataset[i * 10000:(i + 1) * 10000]
        model_wise_exp = exp_dataset[i * 10000:(i + 1) * 10000]
        model_wise_feat = gradcam[i * 10000:(i + 1) * 10000]
        feat_aug_rot, feat_aug_rot_label, feat_aug_rot_exp = _get_rot_aug_dataset(model_wise_traj,
                                                                                  model_wise_label,
                                                                                  model_wise_exp, model_wise_feat,
                                                                                  percentile, aug_size)
        aug_traj_dataset.extend(feat_aug_rot)
        aug_label_dataset.extend(feat_aug_rot_label)
        aug_exp_dataset.extend(feat_aug_rot_exp)

    aug_dataset = [aug_traj_dataset, aug_label_dataset, aug_exp_dataset]
    np.save(save_path, np.array(aug_dataset, dtype=object))

    return


################# Result Part - noiseless ###################


def _save_augmentation_results(ps=[60, 100]):
    for opt, p in zip(["grad","rand"], ps):
        total_table_path = f"./analysis_results/augmentation_results/p{p}_augmentation_results_noiseless.csv"
        if os.path.exists(total_table_path):
            total_table = pd.read_csv(total_table_path)
        else:
            test_dataset = np.load(f"./dataset_noiseless/test/0.npy", allow_pickle=True)
            acc_total_dict = {"acc": [], "noise": []}
            for noise in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                X, y = _preprocessing_dataset_noising(test_dataset, noise)
                noised_dataloader = _get_dataloader(X,y)
                for i in range(5):
                    model_name = f"resnet18_8_b64_lr0.0001_aug_{opt}_small_{ps[0]}_noiseless"
                    model = resnet18_8()
                    model.load_state_dict(torch.load("./model_backups/"+model_name+f"/checkpoint{i}.pt"), 
                                          strict=False)
                    model.eval()
                    
                    model_results, ground_truth = _evaluate_model(model=model,
                                                                  dataloader=noised_dataloader, 
                                                                  device="cuda:1")
                    acc_total = accuracy_score(model_results, ground_truth)
                    # for acc in acc_total:
                    acc_total_dict["acc"].append(acc_total)
                    acc_total_dict["noise"].append(noise)
                    # print(acc_total_dict)
                total_table = pd.DataFrame(acc_total_dict)
            total_table.to_csv(total_table_path)

    return total_table


def _augmentation_results():
    merged_table = {'type': [], "acc": [], "noise": []}
    for p in [60, 100]:
        total_table_path = f"./analysis_results/augmentation_results/p{p}_augmentation_results_noiseless.csv"
        if os.path.exists(total_table_path):
            total_table = pd.read_csv(total_table_path)
            for row in total_table.values:
                merged_table['type'].append(p)
                merged_table['acc'].append(row[1])
                merged_table['noise'].append(row[2])
    merged_table = pd.DataFrame(merged_table)
    compare_table = merged_table[merged_table["type"].isin([60, 100])]
    compare_table["model"] = compare_table["type"]
    compare_table["model"] = compare_table["model"].replace(60, "Targeted augmentation")
    compare_table["model"] = compare_table["model"].replace(100, "Random augmentation")
    compare_table["acc"] = compare_table["acc"] * 100

    # Create main plot
    fig, ax = plt.subplots()
    sns.lineplot(data=compare_table, x="noise", y="acc", hue="model", errorbar="se", palette=['k', 'gray'],
                 style="model", ax=ax)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[0:], labels=labels[0:])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("1/SNR")
    ax.legend(loc="lower left")

    # Dataframe for inset
    
    # Group by noise level and model to get average accuracy
    grouped = compare_table.groupby(["noise", "model"]).mean().reset_index()
    
    # Pivot the table to get differences
    pivot_table = grouped.pivot(index="noise", columns="model", values="acc")
    pivot_table["difference"] = pivot_table["Targeted augmentation"] - pivot_table["Random augmentation"]
    

    # Create inset plot
    ax_inset = inset_axes(ax, width="30%", height="40%", loc="upper right", borderpad=1.5)
    sns.lineplot(data=pivot_table, x=pivot_table.index, y="difference", color='k', marker='o', ms=4, ax=ax_inset)

    ax_inset.set_ylabel(r"$\Delta$ Accuracy (%pt)", size=6)
    ax_inset.set_xlabel("", size=6)
    ax_inset.xaxis.set_tick_params(labelsize=6)
    ax_inset.yaxis.set_tick_params(labelsize=6)
    
    # ax_inset.set_title(r"$\Delta$ Accuracy", size=5)

    plt.savefig("./figures/augmentation_results_noiseless.pdf")
    plt.show()
