import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from utils._preprocessing import _preprocessing_dataset, _get_dataloader
from utils._model_results import _get_model_results, _evaluate_model
from utils._resnet import resnet18_8

from time import time
from itertools import groupby
from operator import itemgetter
import torch

def divide_trajectory(trajectory):
    subtrajectories = []
    current_value = trajectory[0]
    start_index = 0

    for i in range(1, len(trajectory)):
        if trajectory[i] != current_value:
            subtrajectories.append(trajectory[start_index:i])
            start_index = i
            current_value = trajectory[i]

    # Append the last subtrajectory
    subtrajectories.append(trajectory[start_index:])

    return subtrajectories


def divide_trajectory_with_indices(trajectory, divide_indices):
    subtrajectories = []
    start_index = 0

    for idx in divide_indices:
        if len(trajectory)==idx:
            continue
        subtrajectories.append(trajectory[start_index:idx])
        start_index = idx

    # Append the last subtrajectory
    subtrajectories.append(trajectory[start_index:])

    return subtrajectories


def get_subtrajectory_indices(subtrajectories, a, b):
    indices = []
    for i, subtrajectory in enumerate(subtrajectories):
        if np.all(subtrajectory > a) and np.all(subtrajectory < b):
            indices.append(i)
    return indices


def replace_subtrajectories_with_zeros(subtrajectories, indices):
    for idx in indices:
        subtrajectories[idx] = np.zeros_like(subtrajectories[idx])


def replace_subtrajectories_add_noise(subtrajectories, indices, sigma):
    for idx in indices:
        dx = np.random.normal(0, sigma, len(subtrajectories[idx]))
        subtrajectories[idx] = subtrajectories[idx] + dx

def check_subtrajectory_same_value(subtrajectories):
    for subtrajectory in subtrajectories:
        if np.all(subtrajectory == subtrajectory[0]):
            pass
        else:
            return False
    return True



def split_into_consecutive_groups(lst):
    result = []
    current_group = []
    
    for i in range(len(lst)):
        if i == 0 or lst[i] == lst[i-1] + 1:
            current_group.append(lst[i])
        else:
            result.append(current_group)
            current_group = [lst[i]]
    
    # Append the last group
    if current_group:
        result.append(current_group)
    
    return result


def get_random_indexes(lst):
    groups = split_into_consecutive_groups(lst)
    groups_len = [len(group) for group in groups]
    perm_groups_len = np.random.permutation(groups_len)
    selected_indexes = []
    base_list = np.arange(32)
    
    for length in perm_groups_len:
        init_point = np.random.choice(base_list)
        curr_group = list(base_list[init_point:init_point+length])
        while True:
            init_point = np.random.choice(base_list)
            curr_group = list(base_list[init_point:init_point+length])
            # print(curr_group, selected_indexes)
            
            if len(base_list) > init_point + length and not np.isin(curr_group, selected_indexes).any():
                break
        selected_indexes.extend(curr_group)

    return selected_indexes


def _get_datasets_trajwise(dataset, gradcam, p1, p2, random="Pure"):
    X, y = _preprocessing_dataset(dataset)
    X_noised_dataset = []
    X_random_dataset = []

    divide_indices = np.cumsum([len(sub) for sub in divide_trajectory(gradcam[0])])
    for i in (range(len(X))):
        curr_attribution = gradcam[i]
        t1 = np.percentile(curr_attribution, p1)
        t2 = np.percentile(curr_attribution, p2)
        sub_trajectories = divide_trajectory_with_indices(curr_attribution, divide_indices)
        if not check_subtrajectory_same_value(sub_trajectories):
            print("Error")

        X_noised = []
        X_random = []

        targeted_indices = get_subtrajectory_indices(sub_trajectories, t1, t2)

        x_noised = np.copy(X[i][0])
        y_noised = np.copy(X[i][1])

        x_random = np.copy(X[i][0])
        y_random = np.copy(X[i][1])

        x_noised_sub = divide_trajectory_with_indices(x_noised, divide_indices)
        y_noised_sub = divide_trajectory_with_indices(y_noised, divide_indices)
        x_random_sub = divide_trajectory_with_indices(x_random, divide_indices)
        y_random_sub = divide_trajectory_with_indices(y_random, divide_indices)

        replace_subtrajectories_with_zeros(x_noised_sub, targeted_indices)
        replace_subtrajectories_with_zeros(y_noised_sub, targeted_indices)
        
        if random=="Pure":
            random_indicies = np.random.choice(len(x_random_sub), len(targeted_indices), replace=False)
        else:
            random_indicies = get_random_indexes(targeted_indices)
        
        replace_subtrajectories_with_zeros(x_random_sub, random_indicies)
        replace_subtrajectories_with_zeros(y_random_sub, random_indicies)

        # shape (,1000)
        x_noised = np.concatenate(x_noised_sub)
        y_noised = np.concatenate(y_noised_sub)
        x_random = np.concatenate(x_random_sub)
        y_random = np.concatenate(y_random_sub)

        # shape (2,1000)
        X_noised.append(np.stack((x_noised, y_noised)))
        X_random.append(np.stack((x_random, y_random)))

        X_noised_dataset.append(np.array(X_noised))
        X_random_dataset.append(np.array(X_random))
        if i==0:
            plt.plot(x_noised)
            plt.plot(y_noised)
            plt.show()

            plt.plot(x_random)
            plt.plot(y_random)
            plt.show()
            print(targeted_indices, random_indicies)

    # shape ~ ( len, 2, 1000)
    X_noised_dataset = np.squeeze(np.array(X_noised_dataset).astype(np.float32))
    X_random_dataset = np.squeeze(np.array(X_random_dataset).astype(np.float32))
    
    return X_noised_dataset, X_random_dataset, y


def _gradcam_occlusion_results_trajwise(dataset, gradcam, 
                                        model, tag="", 
                                        p1=80, p2=100, random="Pure", noise=False):
    if random=="Pure":
        folder_path = f"./analysis_results/erasing_method/"
    else:
        folder_path = f"./analysis_results/erasing_method/consecutive_conservation"

    if noise==False:
        erased_total_path = folder_path + f"p{p1}_{p2}_erased_total_noiseless.npy"
        random_total_path = folder_path + f"p{p1}_{p2}_random_total_noiseless.npy"
    else:
        erased_total_path = folder_path + f"p{p1}_{p2}_erased_total.npy"
        random_total_path = folder_path + f"p{p1}_{p2}_random_total.npy"
    
    
    if os.path.exists(erased_total_path) and os.path.exists(random_total_path):
        erased_total = np.load(erased_total_path)
        random_total = np.load(random_total_path)
    else:
        erased_total = []
        random_total = []
        noised = []
        random = []
        results_total = []

        start = time()

        X_erased, X_random, y = _get_datasets_trajwise(dataset, gradcam, p1=p1, p2=p2, random=random)
        print("Dataset configuration DONE")

        erased_ds = _get_dataloader(X_erased, y)
        random_ds = _get_dataloader(X_random, y)

        erased_results, ground_truth = _evaluate_model(model, erased_ds,"cuda")
        erased_acc = accuracy_score(erased_results, ground_truth)
        
        random_results, ground_truth = _evaluate_model(model, random_ds, "cuda")
        random_acc = accuracy_score(random_results, ground_truth)

        noised.append(erased_acc)
        random.append(random_acc)

        results_total.append(np.array([erased_results, random_results, ground_truth]))

        erased_total.append(erased)
        random_total.append(random)

        np.save(erased_total_path, np.array(erased_total))
        np.save(random_total_path, np.array(random_total))

    return erased_total, random_total


def load_data(p1s, p2s, folder_path, filename_pattern):
    data = []
    for p1, p2 in zip(p1s, p2s):
        try:
            data.append(np.load(os.path.join(folder_path, filename_pattern.format(p1, p2))))
        except FileNotFoundError:
            print(f"No data found for p1={p1}, p2={p2}. Quitting...")
            return None
    return data


def plot_erasing_results(random_results, erased_results, noise=False):
    if noise==False:
        save_path = "./figures/erasing_method_results_noiseless.pdf"
    else:
        save_path = "./figures/erasing_method_results.pdf"

    plt.ylabel(r"Accuracy (%)")
    plt.xlabel("Percentile")
    x = [10*i for i in range(len(random_results))]
    y = np.array(erased_results).flatten() * 100
    bar_width = 10
    plt.hlines(np.mean(np.array(random_results).flatten()) * 100, xmin=-10, xmax=110, label="Random", ls="--",
               color="gray")
    plt.bar(x, y, width=bar_width, label="Grad-CAM based", color="red", alpha=0.5)
    x = [10*i for i in range(11)]
    # Adjust x-ticks to align with the center of each bar

    plt.xticks(np.array(x) - bar_width / 2, [f"{tick}" for tick in x])
    plt.xlim(-10, 100)
    plt.ylim(np.min(np.array(erased_results).flatten() * 100)-1, np.max(np.array(erased_results).flatten() * 100)+1)
    plt.grid(False)
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def _erasing_method_vis(p1s, p2s, random="Pure", noise=False):
    if random=="Pure":
        folder_path = f"./analysis_results/erasing_method/"
    else:
        folder_path = f"./analysis_results/erasing_method/consecutive_conservation"

    if noise==False:
        file_pattern_random = "p{}_{}_random_total_noiseless.npy"
        file_pattern_erased = "p{}_{}_erased_total_noiseless.npy"
    else:
        file_pattern_random = "p{}_{}_random_total.npy"
        file_pattern_erased = "p{}_{}_erased_total.npy"
        
    random_results = load_data(p1s, p2s, folder_path, file_pattern_random)
    erased_results = load_data(p1s, p2s, folder_path, file_pattern_erased)

    if random_results is None or erased_results is None:
        return

    plot_erasing_results(random_results, erased_results)
