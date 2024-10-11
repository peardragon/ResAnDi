import torch
import numpy as np
import os
from tqdm import tqdm
from utils._resnet import resnet18_8
from utils._preprocessing import _preprocessing
from captum.attr import LayerGradCam
from captum.attr import LayerAttribution


def _save_dataset_gradcam(dataset_path, 
                          model_path = f"./model_backups/resnet18_8_b64_lr0.0001/checkpoint.pt",
                          tag="", shuffle=False, device='cuda'):

    save_dir = f"./Grad-CAM/GradCAM-Residual-{tag}.npy"
    if os.path.exists(save_dir):
        return
    dataset = np.load(dataset_path, allow_pickle=True)

    model = resnet18_8()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    dataloader = _preprocessing(dataset, batch_size=256, shuffle=shuffle)

    model.to(device)
    attributions = []
    sampled_dim = 1000

    with torch.no_grad():
        for dataset in tqdm(dataloader):
            (X,y) = dataset
            inputs = X.to(device)
            target = y.to(device)

            gradcam = LayerGradCam(model, model.layer4)
            attribution = gradcam.attribute(inputs=inputs, target=target)
            attribution = torch.squeeze(attribution, -2)
            attribution = LayerAttribution.interpolate(attribution, sampled_dim, interpolate_mode="nearest")
            attributions.extend(torch.squeeze(attribution).tolist())

    np.save(save_dir, np.array(attributions))
    torch.cuda.empty_cache()

    return


def _save_dataset_gradcam_raw(dataset_path,
                              model_path = f"./model_backups/resnet18_8_b64_lr0.0001/checkpoint.pt",
                              tag="", shuffle=False, device='cuda'):

    save_dir = f"./Grad-CAM/GradCAM-raw-{tag}.npy"
    if os.path.exists(save_dir):
        return
    dataset = np.load(dataset_path, allow_pickle=True)

    model = resnet18_8()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    dataloader = _preprocessing(dataset, batch_size=256, shuffle=shuffle)

    model.to(device)
    attributions = []

    with torch.no_grad():
        for dataset in tqdm(dataloader):
            (X,y) = dataset
            inputs = X.to(device)
            target = y.to(device)
            gradcam = LayerGradCam(model, model.layer4)
            attribution = gradcam.attribute(inputs=inputs, target=target)
            attribution = torch.squeeze(attribution, -2)
            attributions.extend(torch.squeeze(attribution).tolist())

    np.save(save_dir, np.array(attributions))
    torch.cuda.empty_cache()

    return