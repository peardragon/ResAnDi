import os

import numpy as np
import pandas as pd
import scipy.io as sci
import torch
from tqdm import tqdm

from sklearn.manifold import TSNE

from utils._resnet import resnet18_8
from utils._preprocessing import _preprocessing

import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt


##### FEATURE EXTRACTION LOOP
def feature_saving(dataloaders, model):

    save_dir = f"./analysis_results/tsne_results/features/"
    if os.path.exists(save_dir + "ground_truth.npy"):
        print("Exist")
        return
    # placeholders
    PREDS = []
    FEATS1 = []
    FEATS2 = []
    FEATS3 = []
    FEATS4 = []
    FC = []
    model_results = []
    ground_truth = []
    # placeholder for batch features
    features = {}
    
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
    
        return hook
    
    avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    
    ##### REGISTER HOOK
    model.layer1.register_forward_hook(get_features('feat1'))
    model.layer2.register_forward_hook(get_features('feat2'))
    model.layer3.register_forward_hook(get_features('feat3'))
    model.layer4.register_forward_hook(get_features('feat4'))
    model.avgpool.register_forward_hook(get_features('avgpool'))
    
    model.to('cuda')
    # loop through batches
    with torch.no_grad():
        for dataloader in dataloaders:
            for dataset in tqdm(dataloader):
                (X, y) = dataset
                inputs = X.to('cuda')
                results = model(inputs)
                results_label = torch.argmax(results, dim=-1)
                model_results.extend(torch.squeeze(results_label).tolist())
                ground_truth.extend(torch.squeeze(y).tolist())
    
                # add feats and preds to lists
                feat1 = avgpool(features['feat1']).cpu().numpy()
                FEATS1.extend(feat1)
    
                feat2 = avgpool(features['feat2']).cpu().numpy()
                FEATS2.extend(feat2)
    
                feat3 = avgpool(features['feat3']).cpu().numpy()
                FEATS3.extend(feat3)
    
                feat4 = avgpool(features['feat4']).cpu().numpy()
                FEATS4.extend(feat4)
    
                FC.extend(features['avgpool'].cpu().numpy())
    
    
    with open(save_dir + "feature1.npy", 'wb') as f:
        np.save(f, np.array(FEATS1))
    with open(save_dir + "feature2.npy", 'wb') as f:
        np.save(f, np.array(FEATS2))
    with open(save_dir + "feature3.npy", 'wb') as f:
        np.save(f, np.array(FEATS3))
    with open(save_dir + "feature4.npy", 'wb') as f:
        np.save(f, np.array(FEATS4))
    with open(save_dir + "fc.npy", 'wb') as f:
        np.save(f, np.array(FC))
    
    with open(save_dir + "model_results.npy", 'wb') as f:
        np.save(f, np.array(model_results))
    with open(save_dir + "ground_truth.npy", 'wb') as f:
        np.save(f, np.array(ground_truth))

def get_tsne_results(layer_num, wrong_pred=False, verbose=False):


    if os.path.exists(f"./analysis_results/tsne_results/X{layer_num}.npy"):
        print("Exist")
        X_hat = np.load(f"./analysis_results/tsne_results/X{layer_num}.npy")
    else:
        feat = np.load(f"./analysis_results/tsne_results/features/feature{layer_num}.npy", allow_pickle=True)
        feat = np.squeeze(feat)
        tsne = TSNE(n_components=2)
        X_hat = tsne.fit_transform(feat)
        np.save(f"./analysis_results/tsne_results/X{layer_num}.npy", X_hat)

    truth = np.load(f"./analysis_results/tsne_results/features/ground_truth.npy")
    preds = np.load(f"./analysis_results/tsne_results/features/model_results.npy")


    label_list = ["Sub ATTM", "Sub CTRW", "Sub FBM", "Sub SBM", "Sup FBM", "Sup LW", "Sup SBM", "STD BM"]
    color_dict = {0: 'C0', 1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6", 7: "C7"}
    label_dict = {key: values for key, values in enumerate(label_list)}
    legend_details = [{'label': label, 'color': f"C{int(color)}"} for color, label in enumerate(label_list)]
    mapped_values = np.vectorize(color_dict.get)(truth)
    mapped_labels = np.vectorize(label_dict.get)(truth)

    if verbose:
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter1 = ax.scatter(
            X_hat[:, 0], X_hat[:, 1],
            c=[d['color'] for label in mapped_labels for d in legend_details if d['label'] == label],
            s=0.5
        )
    
        handles = [
            plt.Line2D(
                [0], [0],
                marker='o', color='w',
                markerfacecolor=d['color'],
                markersize=10,
                label=d['label']
            ) for d in legend_details
        ]
        labels = [d['label'] for d in legend_details]
    
        if wrong_pred:
            scatter2 = ax.scatter(
                X_hat[:, 0][truth != preds], X_hat[:, 1][truth != preds],
                c='k',
                s=0.5,
                alpha=0.2,
                label="Wrong Predictions"
            )
    
        plt.grid(False)
        plt.axis(False)
        plt.savefig(f"./figures/tsne_{layer_num}.jpg")
        plt.show()

