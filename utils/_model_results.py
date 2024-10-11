import os
import numpy as np
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from utils._preprocessing import _preprocessing
from utils._resnet import resnet18_8

label_list = ["Sub ATTM", "Sub CTRW", "Sub FBM", "Sub SBM", "Sup FBM", "Sup LW", "Sup SBM", "STD BM"]


def _save_model_results(model_name, save_tag, model_results, ground_truth):
    model_results_path = f"./analysis_results/model_results/{model_name}_model_results.npy"
    ground_truth_path = f"./analysis_results/model_results/{model_name}_ground_truth.npy"
    
    np.save(model_results_path, model_results)
    np.save(ground_truth_path, ground_truth)


def _evaluate_model(model, dataloader, device):
    model.to(device)
    model_results = []
    ground_truth = []

    with torch.no_grad():
        for dataset in tqdm(dataloader):
            (X, y) = dataset
            inputs = X.to(device)
            results = model(inputs)
            softmax_output = torch.nn.Softmax(dim=-1)(results).tolist()
            results_label = torch.argmax(results, dim=-1)
            model_results.extend(torch.squeeze(results_label).tolist())
            ground_truth.extend(torch.squeeze(y).tolist())

    return model_results, ground_truth


def _plot_confusion_matrix(ground_truth, model_results):
    acc = accuracy_score(ground_truth, model_results)
    print("Accuracy: ", acc)
    confusion = confusion_matrix(ground_truth, model_results, normalize="true")
    confusion_df = pd.DataFrame(confusion)
    confusion_df.columns = label_list
    confusion_df.index = label_list

    ax = sns.heatmap(confusion_df, annot=True, square=True, fmt=".1%",
                      cmap=sns.color_palette("vlag", as_cmap=True),
                     annot_kws={"size": 5},
                     )
    plt.ylabel("Ground Truth")
    plt.xlabel("Predicted Label")
    plt.savefig(f"./figures/_model_results_confusion_plot.pdf", bbox_inches='tight')
    plt.show()



def _get_model_results(dataset=None, model_name="resnet18_8_b64_lr0.0001", model_tag="", shuffle=False,
                       device="cuda", verbose=True, save_tag="", load=True, save=False):
    if load:
        model_results_path = f"./analysis_results/model_results/{model_name}_model_results.npy"
        ground_truth_path = f"./analysis_results/model_results/{model_name}_ground_truth.npy"

        if os.path.exists(model_results_path) and os.path.exists(ground_truth_path):
            model_results = np.load(model_results_path, allow_pickle=True)
            ground_truth = np.load(ground_truth_path, allow_pickle=True)

            if verbose:
                _plot_confusion_matrix(ground_truth, model_results)
            return model_results, ground_truth
            
        else:
            pass

        
    model_path = f"./model_backups/{model_name}/checkpoint{model_tag}.pt"
    if os.path.exists(model_path):
        model = resnet18_8()
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()

        dataloader = _preprocessing(dataset, batch_size=256, shuffle=shuffle)
        model_results, ground_truth = _evaluate_model(model, dataloader, device)

        if save:
            _save_model_results(model_name, save_tag, model_results, ground_truth)

    if verbose:
        _plot_confusion_matrix(ground_truth, model_results)

    torch.cuda.empty_cache()

    return model_results, ground_truth


def _model_confidence_plot(model_results, dataset):
    exponents = dataset[-1].flatten()
    ground_truth = dataset[1].flatten()
    exp_ranges = np.arange(0.1, 2+0.05, 0.05)
    
    df = pd.DataFrame(np.array([exponents, ground_truth, model_results]).T,
                 columns=["Exponent", "Truth Label", "Predicted Label"])

    df["Result"] = (df["Truth Label"]==df["Predicted Label"])
    df['Truth Label'] = df['Truth Label'].astype(int)

    grouped_df = pd.DataFrame( df.groupby(['Exponent', "Truth Label"])["Result"].sum()).reset_index(level=[0,1])
    df['range'] = df.groupby('Truth Label')[['Exponent']].transform(lambda x: pd.cut(x, bins = exp_ranges))
    df["Exponent"] = df["range"].apply(lambda x: x.mid)
    
    
    predicted_counts = df.groupby(['Exponent', "Truth Label","Predicted Label"])["Predicted Label"].count()
    total_counts = df.groupby(['Exponent', "Truth Label"])["Truth Label"].count()

    exponents_confidence_df = pd.DataFrame(predicted_counts/total_counts, columns=["Confidence"])
    exponents_confidence_df = exponents_confidence_df.reset_index(level=[0,1,2])
    
    total_list = []
    label_num = 8 
    for truth_label in range(label_num):
        df = exponents_confidence_df[exponents_confidence_df["Truth Label"]==truth_label]
        sub_list = []
        for pred_label in range(label_num):
            confidence = df[df["Predicted Label"]==pred_label]["Confidence"].values
            exp_index = df[df["Predicted Label"]==pred_label]["Exponent"].values
            sub_table = dict(zip(exp_index, confidence))

            exp_default = df[df["Predicted Label"]==df["Truth Label"]]["Exponent"].values

            for exp in exp_default:
                if exp not in sub_table.keys():
                    sub_table[exp]=0
            sub_table = dict(sorted(sub_table.items()))
            sub_list.append(sub_table)
        total_list.append(sub_list)

    # Plot
    sns.set_palette("deep")
    sns.set(font_scale = 2)
    sns.set_style("whitegrid", {'axes.grid': False})
    fig, ax = plt.subplots(2, label_num//2, figsize=(5*label_num,20), sharey=True)
    i=0
    for datas in total_list:
        if i==label_num-1:
            x = [1.25, 1.875]
            y = [[list(datas[i].values())[17], list(datas[i].values())[17]] for i in range(label_num)]
            ax[i//4, i%4].stackplot(x,y)
            ax[i//4, i%4].set_title(f'Ground Truth: {label_list[i]}')

            ax[i//4, i%4].set_xlabel(r"Diffusion Exponent $\alpha$ = 1")
            # ax[i//4, i%4].set_xlim(1.1, 1.9)
            ax[i//4, i%4].set_xticklabels([])
            ax[i//4, i%4].margins(0,0)
            ax[i//4, i%4].set_xmargin(0)
            ax[i//4, i%4].spines[['right','top','left','bottom']].set_visible(False)


        else:
            x = list(datas[0].keys())
            y = [list(datas[i].values()) for i in range(label_num)]
            ax[i//4, i%4].set_title(f'Ground Truth: {label_list[i]}')
            ax[i//4, i%4].stackplot(x,y)
            ax[i//4, i%4].set_xlabel(r"Diffusion Exponent $\alpha$")
            ax[i//4, i%4].set_xlim(1.1 if i > 3 else 0.1, 1.9 if i > 3 else 0.9)
            ax[i//4, i%4].margins(0,0)
            ax[i//4, i%4].set_xmargin(0)
            ax[i//4, i%4].spines[['right','top','left','bottom']].set_visible(False)
            i+=1
    # plt.tight_layout()
    # fig.supylabel("Confidence Level")
    # fig.suptitle("Confidence Prediction")
    # fig.legend(labels=label_list)

    plt.savefig("./figures/confidence_level.svg")

    plt.show()

