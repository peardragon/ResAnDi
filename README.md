# ResAnDi: Residual Network Analysis for Anomalous Diffusion

This project implements a comprehensive analysis framework for studying anomalous diffusion phenomena using Residual Neural Networks (ResNets). The framework includes data generation, model training, interpretability analysis through Grad-CAM, statistical relevance testing, and various visualization techniques.

## Project Overview

ResAnDi provides a complete pipeline for:
- **Synthetic trajectory generation** for different stochastic processes (Brownian motion, fractional Brownian motion, etc.)
- **Deep learning classification** using custom 1D ResNet architectures
- **Model interpretability** through Gradient-based Class Activation Mapping (Grad-CAM)
- **Statistical relevance analysis** and feature importance evaluation
- **Data augmentation** and robustness testing
- **Dimensionality reduction** visualization using t-SNE
- **Feature ablation studies** and blockwise analysis

## Installation

Install the required dependencies using uv:

```bash
uv pip install -r requirements.txt
```

Refer [SETUP.md](./SETUP.md) for detail.

## Playground

**`playground.ipynb`** notebook provides a simplified demonstration of the ResAnDi workflow. It includes:
1. **Dataset Generation** - Create synthetic trajectories from diffusion models
2. **Trajectory Sample Visualization** - Visualize sample trajectories
3. **Data Preprocessing** - Prepare data for training
4. **Model Training** - Train ResNet classifier on trajectory data
5. **Grad-CAM Analysis** - Generate attribution maps
6. **Visualization** - Plot trajectories with Grad-CAM overlays
7. **t-SNE Analysis** - Dimensionality reduction and clustering visualization


## Directory Structure

```
ResAnDi/
├── utils/                          # Core utility modules
│   ├── _dataset_generation.py      # Synthetic trajectory generation
│   ├── _resnet.py                  # Custom 2D ResNet architecture
│   ├── _train.py                   # Training utilities and early stopping
│   ├── _preprocessing.py           # Data preprocessing and DataLoader creation
│   ├── _gradcam.py                 # Grad-CAM implementation
│   ├── _model_results.py           # Model evaluation and results analysis
│   ├── _statistical_relevance.py   # Statistical significance testing
│   ├── _erasing_method.py          # Feature erasing and importance analysis
│   ├── _augmentation.py            # Data augmentation techniques
│   ├── _tsne_analysis.py           # t-SNE visualization
│   ├── _example_traj.py            # Trajectory visualization utilities
│   └── _noising.py                 # Noise injection utilities
│
├── dataset_noiseless/              # Generated datasets
│   ├── train/                      # Training datasets (varying length)
│   ├── test/                       # Test datasets (varying length)
│   ├── eval/                       # Evaluation datasets (varying length)
│   ├── train_1000/                 # Fixed-length training datasets (1000 timesteps)
│   ├── test_1000/                  # Fixed-length test datasets (1000 timesteps)
│   └── eval_1000/                  # Fixed-length evaluation datasets (1000 timesteps)
│
├── Grad-CAM/                       # Grad-CAM attribution maps
│   ├── GradCAM-Residual-*.npy      # Processed attribution maps (interpolated)
│   └── GradCAM-raw-*.npy           # Raw attribution maps
│
├── model_ckpt/                     # Model checkpoints during training
├── saved_models/                   # Final trained model weights
├── analysis_results/               # Analysis outputs
│   ├── model_results/              # Model performance metrics
│   ├── statistical_relevance/      # Statistical significance results
│   ├── erasing_method/             # Feature erasing analysis
│   ├── augmentation_results/       # Data augmentation results
│   ├── feature_ablation/           # Feature ablation study results
│   ├── tsne_results/               # t-SNE visualization results
│   └── statistical_relevance_block/# Block-wise statistical analysis
│
└── figures/                        # Generated plots and visualizations
    └── *.pdf/*.svg                 # Various analysis plots
```

## Notebooks Workflow & Dependencies

### 1. Data Generation & Preprocessing
- **`__make_dataset-noiseless.ipynb`**: Generate synthetic trajectory datasets
  - **Utils Dependencies**: `_dataset_generation.dataset_generation`
  - **Purpose**: Create noiseless trajectory data for both varying length (10-1000) and fixed length (1000) datasets

- **`__train_procedure-noiseless.ipynb`**: Train ResNet models on generated data
  - **Utils Dependencies**: `_resnet.resnet18_8`, `_preprocessing._preprocessing`, `_train.train`
  - **Purpose**: Train the main classification model using early stopping and validation

### 2. Model Analysis & Results
- **`_0_train_results-noiseless.ipynb`**: Evaluate trained model performance
  - **Utils Dependencies**: `_model_results._get_model_results`, `_model_results._evaluate_model`
  - **Purpose**: Generate confusion matrices, accuracy metrics, and performance analysis

- **`_1_gradcam_procedure-noiseless.ipynb`**: Generate Grad-CAM attribution maps
  - **Utils Dependencies**: `_resnet.resnet18_8`, `_preprocessing._preprocessing`, `_gradcam._save_dataset_gradcam`, `_gradcam._save_dataset_gradcam_raw`
  - **Purpose**: Create interpretability maps for model decision analysis

### 3. Significance Analysis
- **`_2_erasing_procedure-noiseless.ipynb`**: Feature erasing methodology
  - **Utils Dependencies**: `_resnet.resnet18_8`, `_gradcam._save_dataset_gradcam`, `_erasing_method._gradcam_occlusion_trajwise`, `_erasing_method._erasing_method_vis`, `_erasing_method.load_data`
  - **Purpose**: Systematic trajectory segment removal to assess feature importance

- **`_2_erasing_results-noiseless.ipynb`**: Analysis of erasing experiment results
  - **Utils Dependencies**: `_erasing_method` (visualization functions)
  - **Purpose**: Visualize and analyze accuracy drops from feature erasing

### 4. Data Augmentation Studies
- **`_3_augmentation_dataset_prepare-noiseless.ipynb`**: Prepare augmented datasets
  - **Utils Dependencies**: `_resnet.resnet18_8`, `_gradcam._save_dataset_gradcam`, `_augmentation._save_aug_dataset_rot`
  - **Purpose**: Generate gradient-based and random augmented training data

- **`_3_augmentation_procedure_training_grad-noiseless.ipynb`**: Gradient-based augmentation training
  - **Utils Dependencies**: `_resnet.resnet18_8`, `_preprocessing._preprocessing`, `_train.train`
  - **Purpose**: Train models using Grad-CAM guided data augmentation

- **`_3_augmentation_procedure_training_rand-noiseless.ipynb`**: Random augmentation training
  - **Utils Dependencies**: `_resnet.resnet18_8`, `_preprocessing._preprocessing`, `_train.train`
  - **Purpose**: Train models using random noise-based augmentation

- **`_3_augmentation_results-noiseless.ipynb`**: Compare augmentation strategies
  - **Utils Dependencies**: `_augmentation` (analysis and visualization functions)
  - **Purpose**: Performance comparison between augmentation approaches

### 5. TSNE analysis with fixed length dataset
- **`_4_tsne_procedure_training.ipynb`**: t-SNE on training data
  - **Utils Dependencies**: `_resnet.resnet18_8`, `_preprocessing._preprocessing`, `_train.train`, `_model_results._get_model_results`
  - **Purpose**: Feature extraction and t-SNE analysis on training features

- **`_4_tsne_procedure.ipynb`**: General t-SNE analysis
  - **Utils Dependencies**: `_resnet.resnet18_8`, `_preprocessing._preprocessing`, `_tsne_analysis.feature_saving`
  - **Purpose**: Extract and save features from different network layers for t-SNE

- **`_4_tsne_results.ipynb`**: t-SNE visualization results
  - **Utils Dependencies**: `_tsne_analysis.feature_saving`, `_tsne_analysis.get_tsne_results`, `_resnet.resnet18_8`
  - **Purpose**: Generate and visualize t-SNE embeddings for feature space analysis

### 6. Statistical Analysis
- **`_5_statistical_relevance_procedure-noiseless.ipynb`**: Statistical significance testing
  - **Utils Dependencies**: 
    - `_gradcam._save_dataset_gradcam_raw`
    - `_preprocessing._preprocessing_dataset`, `_preprocessing._preprocessing_dataset_noising`
    - `_statistical_relevance.array_split`, `_statistical_relevance._save_dataset_features`, `_statistical_relevance._get_pearson_correlation`
  - **Purpose**: Compute correlations between Grad-CAM attributions and trajectory statistical features

- **`_5_statistical_relevance_results-noiseless.ipynb`**: Statistical analysis results
  - **Utils Dependencies**: `_statistical_relevance` (correlation analysis and visualization)
  - **Purpose**: Generate correlation heatmaps and statistical significance plots

- **`_5_statistical_relevance_subtraj_length.ipynb`**: Sub-trajectory length analysis
  - **Utils Dependencies**: `_statistical_relevance._get_conv_response`, `_resnet.resnet18_8`
  - **Purpose**: Analyze how sub-trajectory length affects feature importance

- **`_6_sample_trajs_noiseless.ipynb`**: Sample trajectory visualizations with Grad-CAM
  - **Utils Dependencies**: `_example_traj._draw_all_sample_traj`, `_gradcam._save_dataset_gradcam`
  - **Purpose**: Create trajectory visualizations with attribution overlays

### 7. Feature Ablation Analysis
- **`_7_feature_ablation_procedure.ipynb`**: Feature ablation experiments
  - **Utils Dependencies**: `_preprocessing._preprocessing_dataset`
  - **Purpose**: Systematic removal of individual statistical features to assess contribution

- **`_7_feature_ablation_vis_accuracy_drop.ipynb`**: Accuracy drop visualization
  - **Utils Dependencies**: Feature ablation analysis functions
  - **Purpose**: Visualize model performance degradation from feature removal

- **`_7_feature_ablation_vis_confusion.ipynb`**: Confusion matrix analysis
  - **Utils Dependencies**: Model evaluation utilities
  - **Purpose**: Analyze class-specific impacts of feature ablation

- **`_7_feature_ablation_vis_independence.ipynb`**: Feature independence analysis
  - **Utils Dependencies**: Statistical analysis utilities
  - **Purpose**: Test statistical independence between different trajectory features

### 8. Block-wise Analysis
- **`_8_blockwise_statistical_relevance_B1.ipynb`**: Block 1 statistical analysis
- **`_8_blockwise_statistical_relevance_B2.ipynb`**: Block 2 statistical analysis  
- **`_8_blockwise_statistical_relevance_B3.ipynb`**: Block 3 statistical analysis
  - **Utils Dependencies**: 
    - `_resnet.resnet18_8`
    - `_preprocessing._preprocessing_dataset`, `_preprocessing._get_dataloader`  
    - `_train.train`
  - **Purpose**: Segment-wise analysis of trajectory importance using different network layers

## Utilities Function Reference

### Core Modules

#### `utils/_dataset_generation.py`
- **`dataset_generation()`**: Generate synthetic trajectory datasets for different stochastic processes
- **`regularize()`**: Regularize trajectories with irregular sampling times
- **`datasets_theory()`**: Generate theoretical dataset distributions
- **`_add_noise_with_normalization_scale()`**: Add controlled noise to trajectories

#### `utils/_resnet.py`
- **`resnet18()`**: Standard ResNet18 for 5-class classification  
- **`resnet18_8()`**: ResNet18 modified for 8-class classification
- **`resnet18_1()`**: ResNet18 for binary classification
- **`BasicBlock`**: Residual block implementation with 1×k convolutions
- **`ResNet`**: Main ResNet architecture class with feature map extraction

#### `utils/_preprocessing.py`
- **`_preprocessing()`**: Create DataLoader from trajectory datasets
- **`_preprocessing_dataset()`**: Dataset preprocessing and normalization
- **`_preprocessing_traj()`**: Individual trajectory preprocessing
- **`_get_dataloader()`**: Create DataLoader with specific parameters

#### `utils/_train.py`
- **`train()`**: Main training loop with early stopping
- **`EarlyStopping`**: Early stopping callback class with validation monitoring

### Analysis Modules

#### `utils/_gradcam.py`
- **`_save_dataset_gradcam()`**: Generate and save Grad-CAM attributions with interpolation
- **`_save_dataset_gradcam_raw()`**: Generate and save raw Grad-CAM attributions

#### `utils/_model_results.py`
- **`_get_model_results()`**: Comprehensive model evaluation and metrics
- **`_evaluate_model()`**: Model performance evaluation on test sets
- Functions for confusion matrix generation and accuracy analysis

#### `utils/_statistical_relevance.py`
- **`array_split()`**: Split trajectories into segments for analysis
- **`_save_dataset_features()`**: Extract and save statistical features from trajectories
- **`_get_pearson_correlation()`**: Compute Pearson correlations between features and attributions
- **`_get_pearson_table()`**: Generate correlation tables and significance tests

#### `utils/_erasing_method.py`
- **`_gradcam_occlusion_trajwise()`**: Trajectory-wise occlusion analysis
- **`_erasing_method_vis()`**: Visualization functions for erasing results
- **`load_data()`**: Data loading utilities for erasing experiments
- Functions for systematic feature removal and importance ranking

#### `utils/_augmentation.py`
- **`_save_aug_dataset_rot()`**: Generate rotationally augmented datasets
- **`_dataset_noising()`**: Apply noise-based augmentation
- Functions for gradient-based and random augmentation strategies

#### `utils/_tsne_analysis.py`
- **`feature_saving()`**: Extract and save features (neural networks' internal activations) from different network layers
- **`get_tsne_results()`**: Perform t-SNE analysis and generate embeddings
- Functions for dimensionality reduction visualization

#### `utils/_example_traj.py`
- **`_draw_example_traj()`**: Draw individual trajectory with Grad-CAM overlay
- **`_draw_all_sample_traj()`**: Generate grid of sample trajectories with attributions
- Trajectory visualization utilities with color-coded importance


## Results Storage

- Model checkpoints and final weights are saved in `model_ckpt/` and  `saved_models/`
- Grad-CAM analysis resutls are saved in `Grad-CAM/`
- Analysis results are organized in `analysis_results/` with subdirectories for each analysis type
- Visualizations are saved in `figures/` as PDF and SVG files
