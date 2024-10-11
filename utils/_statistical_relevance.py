import os

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch

from tqdm import tqdm
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler

from utils._preprocessing import _preprocessing_traj, _preprocessing_dataset
from utils._resnet import resnet18_8

import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

###########################################################

######## autocorrelations
def autocorrelation_norm(split_traj_dataset):
    diffs =  np.diff(split_traj_dataset, axis=-1)
    arr_mean = diffs- np.mean(diffs, axis=-1, keepdims=True)
    
    # Compute the autocorrelation for lag 1
    numerator = np.sum(arr_mean[..., :-1] * arr_mean[..., 1:], axis=-1)
    denominator = np.sqrt(np.sum(arr_mean[..., :-1]**2, axis=-1) * np.sum(arr_mean[..., 1:]**2, axis=-1))
    
    # Handle cases where the denominator might be zero to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(denominator != 0, numerator / denominator, 1)
    
    result = np.mean(result, axis=-1)
    return result


###########################################################

# def autocorrelation_norm_stat(split_traj_dataset):
#     diffs_t =  np.diff(split_traj_dataset, axis=-1)[..., :-1]
#     diffs_tp1 =  np.diff(split_traj_dataset, axis=-1)[..., 1:]

#     diff_cov = np.mean(diffs_t * diffs_tp1, axis=-1)
#     diff_mean = np.mean(np.diff(split_traj_dataset, axis=-1),axis=-1)
#     cov = np.var(np.diff(split_traj_dataset, axis=-1), axis=-1)
    
    
#     # Compute the autocorrelation for lag 1
#     numerator = diff_cov - diff_mean**2
#     denominator = cov    
#     # Handle cases where the denominator might be zero to avoid division by zero
#     with np.errstate(divide='ignore', invalid='ignore'):
#         result = np.where(denominator != 0, numerator / denominator, 1)
    
#     result = np.mean(result, axis=-1)
#     return result


def manual_kurtosis(data):
    mean = np.mean(data, axis=-1)
    std_dev = np.std(data, axis=-1)
    fourth_moment = np.mean((data - mean[..., np.newaxis]) ** 4, axis=-1)
    with np.errstate(divide='ignore', invalid='ignore'):
        kurtosis = np.where(std_dev != 0, fourth_moment / (std_dev ** 4) - 3, 0)
    return kurtosis
    
def mean_kurtosis(split_traj_dataset):
    diffs = np.diff(split_traj_dataset, axis=-1)
    
    # Define the slice indices
    div = diffs.shape[-1]/4
    slice_indices = [
        slice(int(div*0), int(div*1)),
        slice(int(div*1), int(div*2)),
        slice(int(div*2), int(div*3)),
        slice(int(div*3), int(div*4))
    ]
    
    # Compute the kurtosis for each slice and take the mean
    kurtosis_means = [np.mean(manual_kurtosis(diffs[..., sl]), axis=-1) for sl in slice_indices]
    # Compute the final result
    result = np.mean(kurtosis_means, axis=0)
    
    return result
    
###########################################################


def jump_event(split_traj_dataset):
    diffs = np.diff(split_traj_dataset, axis=-1)
    increment_ratio = (np.abs(diffs[...,1:])+1e-6)/(np.abs(diffs[...,:-1]+1e-6))
    result = np.max(np.std(increment_ratio, axis=-1),axis=-1)
 
    return result

###########################################################


def varying_diffusivity(split_traj_dataset):
    diffs = np.diff(split_traj_dataset, axis=-1)
    div = diffs.shape[-1]/4

    num = np.mean( - np.std(diffs[..., int(div*0):int(div*1)], axis=-1) 
                   + np.std(diffs[..., int(div*3):int(div*4)], axis=-1),axis=-1)/4
    den = np.mean(np.std(diffs, axis=-1), axis=-1)
        # Handle cases where the denominator might be zero to avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(den != 0, num / den, 0)

    return result

###########################################################

def var_cos_angle(split_traj_dataset):
    diffs = np.diff(split_traj_dataset, axis=-1)
    vx = diffs[:,:,0,:]
    vy = diffs[:,:,1,:]
    # np dot
    num = vx[:,:,:-1]*vx[:,:,1:] + vy[:,:,:-1]*vy[:,:,1:]
    den = np.sqrt((vx[:,:,:-1]**2+vy[:,:,:-1]**2)*(vx[:,:,1:]**2+vy[:,:,1:]**2))
    with np.errstate(divide='ignore', invalid='ignore'):
        turning_angles = np.arccos(num/den)/np.pi
    
    turning_angles = np.nan_to_num(turning_angles, 0)

    result = np.std(turning_angles, axis=-1)
    return result
    
###########################################################

def array_split(trajs):
    # trajs shape (80000, 2, 1000)
    splits = []
    for i in range(32):
        start = 0 * i + 25 * i
        end = 225 + 25 * i
        splits.append(trajs[:, :, start:end])
    return np.stack(splits, axis=1)  # shape (80000, 32, 2, 225)
    


def _save_dataset_features(split_traj_dataset, tag):
    folder_path = f"./analysis_results/statistical_relevance/features"
        
    if os.path.exists(folder_path + f"/f1{tag}.npy"):
        pass
    else:
        f1 = autocorrelation_norm(split_traj_dataset)
        np.save(folder_path + f"/f1{tag}.npy", f1)
        
    if os.path.exists(folder_path + f"/f2{tag}.npy"):
        pass
    else:
        f2 = mean_kurtosis(split_traj_dataset)
        np.save(folder_path + f"/f2{tag}.npy", f2)

    if os.path.exists(folder_path + f"/f3{tag}.npy"):
        pass
    else:
        f3 = jump_event(split_traj_dataset)
        np.save(folder_path + f"/f3{tag}.npy", f3)
        
    if os.path.exists(folder_path + f"/f4{tag}.npy"):
        pass
    else:
        f4 = varying_diffusivity(split_traj_dataset)
        np.save(folder_path + f"/f4{tag}.npy", f4)

    if os.path.exists(folder_path + f"/f5{tag}.npy"):
        pass
    else:
        f5 = var_cos_angle(split_traj_dataset)
        np.save(folder_path + f"/f5{tag}.npy", f5)


def _get_pearson_correlation(tag="train"):
    folder_path = f"./analysis_results/statistical_relevance/features"
    save_path = f"./analysis_results/statistical_relevance/correlation_table_{tag}.npy"

    if os.path.exists(save_path):
        return
    
    # # # tot_feature_prob ~ all train dataset concat
    tot_n1 = [[] for _ in range(8)]
    tot_n2 = [[] for _ in range(8)]
    tot_n3 = [[] for _ in range(8)]
    tot_n4 = [[] for _ in range(8)]
    tot_gc = [[] for _ in range(8)]
    for i in range(5):
        # autocorr
        f1 = np.load(folder_path + f"/f1_{tag}_{i}.npy", allow_pickle=True)
        # kurtosis
        f2 = np.load(folder_path + f"/f2_{tag}_{i}.npy", allow_pickle=True)
        # jump
        f3 = np.load(folder_path + f"/f3_{tag}_{i}.npy", allow_pickle=True)
        # diffusivity
        f4 = np.load(folder_path + f"/f4_{tag}_{i}.npy", allow_pickle=True)
        # consistency
        f5 = np.load(folder_path + f"/f5_{tag}_{i}.npy", allow_pickle=True)
        
        f2 = np.abs(f2)
        f2 = MinMaxScaler().fit_transform(f2.T).T
        f3 = np.abs(f3)
        f3 = MinMaxScaler().fit_transform(f3.T).T
        f5 = np.abs(f5)
        f5 = MinMaxScaler().fit_transform(f5.T).T
        
        gc = np.load(f"./Grad-CAM/GradCAM-raw-{tag}_{i}_1000.npy")
        for i in range(8):
            ###
    
            tot_n1[i].append(f1[i * 10000:(i + 1) * 10000])
            tot_n2[i].append(f1[i * 10000:(i + 1) * 10000] * f5[i * 10000:(i + 1) * 10000])
            tot_n3[i].append(f3[i * 10000:(i + 1) * 10000] * f2[i * 10000:(i + 1) * 10000])
            tot_n4[i].append(f4[i * 10000:(i + 1) * 10000] * f2[i * 10000:(i + 1) * 10000])
            tot_gc[i].append(gc[i * 10000:(i + 1) * 10000])
    

    pearson_table = {"label": [], "opts": [], "pearsonr": [], "spearman": []}
    label_list = ["Sub ATTM", "Sub CTRW", "Sub FBM", "Sub SBM", "Sup FBM", "Sup LW", "Sup SBM", "STD BM"]

    for label in range(8):
        # concat_gc_raw = np.array(tot_gc[label])
        concat_gc_raw = np.concatenate(np.array(tot_gc[label]), axis=0).flatten()
        
        for n, tot_feat in enumerate([tot_n1, tot_n2, tot_n3, tot_n4]):
            # concat_feat = np.array(tot_feat[label])
            concat_feat = np.concatenate(np.array(tot_feat[label]), axis=0).flatten()

            print(concat_feat.shape, concat_gc_raw.shape, n)
            index = ~np.isnan(concat_feat)
            concat_feat = concat_feat[index]
            concat_gc = concat_gc_raw[index]
            pearson_r = pearsonr(concat_gc, concat_feat)
            spearman_r = spearmanr(concat_gc, concat_feat)

            pearson_table["label"].append(label_list[label])
            pearson_table["opts"].append(n)
            pearson_table["pearsonr"].append(pearson_r[0])
            pearson_table["spearman"].append(spearman_r[0])
    np.save(save_path, pearson_table)

def _get_pearson_table(tag="train"):
    table_path = f"./analysis_results/statistical_relevance/correlation_table_{tag}.npy"
    save_path = f"./figures/statistical_relevance_{tag}.svg"


    pearson_table = np.load(table_path, allow_pickle=True)
    
    pd_pearson = pd.DataFrame(dict(pearson_table.tolist()))
    heatmap_data = pd_pearson.pivot(columns='label', index='opts', values='pearsonr')
    
    # Define the colormap with a wide plateau around 0
    colors = [(0, 0, 1), (1, 1, 1), ( 1, 1, 1), (1, 1, 1), (1, 0, 0)]  # Blue to Gray to White to Gray to Red
    n_bins = 100  # Number of bins
    cmap_name = "custom_cmap"
    color_stops = [0.0, 0.85 / 2, 0.5, 1.15 / 2, 1.0]
    custom_cmap = LinearSegmentedColormap.from_list(cmap_name, list(zip(color_stops, colors)), N=n_bins)
    norm = mcolors.Normalize(vmin=-0.4, vmax=0.4)
        
    ax = sns.heatmap(heatmap_data, annot=True, vmin=-0.4, vmax=0.4, cmap=custom_cmap, norm=norm, linewidths=0.1,
                linecolor='white', fmt='.2f', cbar=True, annot_kws={"size": 6})
    # plt.title("Grad-CAM pearson correlation with statistical property")
    ax.set_xticklabels(["BM", "SubATTM", "SubCTRW", "SubFBM", "SubSBM", "SupFBM", "SupLW", "SupSBM"])
    ax.set_yticklabels(ax.get_yticklabels(), rotation="horizontal", horizontalalignment="center")
    ax.set_yticklabels(["AC", "CS", "SG", "VD"])
    plt.xlabel("")
    plt.ylabel("")
    plt.savefig(save_path)
    plt.show()