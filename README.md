# Spatiotemporal Vegetation Coverage Prediction System based on ConvLSTM-UNet

## Project Overview

This project uses a deep learning model (ConvLSTM-UNet) to perform spatiotemporal prediction on remote sensing time-series data, enabling prediction of continuous values for the entire image of the next year given vegetation coverage data from the past several years. Supports multiple vegetation index types: EVI, FVC, NDVI, RVI, RSEI, SAVI.

## Project Structure

```
.
├── data/
│   ├── raw_data/          # Raw TIF data
│   │   ├── EVI矿/
│   │   ├── EVI缓冲区/
│   │   ├── FVC矿/
│   │   ├── FVC缓冲区/
│   │   └── ... (other types)
│   └── interim/           # Intermediate processed data
│       ├── npz/           # Processed NPZ remote sensing data files
│       ├── buffer_mask.pkl  # Buffer zone mask
│       └── mine_mask.pkl    # Mining area mask
├── src/
│   ├── extract_mask.py           # Mask extraction script
│   ├── process_tif_to_npz.py    # TIF to NPZ conversion script
│   ├── train_convlstm_unet.py    # Model training script
│   ├── read_tif.py              # TIF reading utility
│   └── show_npz.py              # NPZ visualization utility
├── models/                # Saved model weights
├── outputs/               # Output results
│   └── plots/            # Visualization charts
└── README.md
```

## Requirements

### Python Version

- Python 3.7+

### Main Dependencies

```
torch >= 1.9.0
numpy >= 1.19.0
matplotlib >= 3.3.0
scikit-learn >= 0.24.0
scipy >= 1.5.0
rasterio >= 1.2.0
tqdm >= 4.60.0
```

### Install Dependencies

```bash
pip install torch numpy matplotlib scikit-learn scipy rasterio tqdm
```

## Dataset

The dataset can be accessed at: [Google Drive](https://drive.google.com/drive/folders/1y0aUqFcft0tMep8wP2riSjihF3G93KDl?usp=sharing)

## Getting Started

### Execution Order

The project must be executed in the following order:

#### 1. Extract Masks

Run `extract_mask.py` to generate mask files:

```bash
python src/extract_mask.py
```

- Reads TIF files from `data/raw_data/EVI缓冲区/` and `data/raw_data/EVI矿/`
- Extracts regions with value 0 as masks
- Saves masks to `data/interim/buffer_mask.pkl` and `data/interim/mine_mask.pkl`

#### 2. Process TIF Data

Run `process_tif_to_npz.py` to convert TIF files to NPZ format:

```bash
python src/process_tif_to_npz.py
```

- Traverses all subdirectories under `data/raw_data/`
- Combines time-series TIF files from each subdirectory into 3D arrays
- Applies masks to filter invalid regions
- Saves as NPZ files to `data/interim/npz/`

#### 3. Train Model

Run `train_convlstm_unet.py` to train the model:

```bash
python src/train_convlstm_unet.py --map_type FVC
```

**Supported map_type parameters:**

- `EVI`
- `FVC`
- `NDVI`
- `RVI`
- `RSEI`
- `SAVI`

## Usage

### Training Script Parameters

`train_convlstm_unet.py` supports the following command-line arguments:

| Parameter                 | Type  | Default            | Description                                      |
| ------------------------- | ----- | ------------------ | ------------------------------------------------ |
| `--map_type`            | str   | **Required** | Index type to train (EVI/FVC/NDVI/RVI/RSEI/SAVI) |
| `--window_size`         | int   | 8                  | Input time window size (years)                   |
| `--patch_size`          | int   | 64                 | Spatial patch size for training                  |
| `--batch_size`          | int   | 128                | Batch size                                       |
| `--num_epochs`          | int   | 100                | Number of training epochs                        |
| `--base_lr`             | float | 0.001              | Base learning rate                               |
| `--num_workers`         | int   | 2                  | Number of data loading workers                   |
| `--early_stop_patience` | int   | 10                 | Early stopping patience                          |
| `--train_samples`       | int   | 20000              | Number of training samples                       |
| `--val_samples`         | int   | 4000               | Number of validation samples                     |
| `--data_dir`            | str   | data               | Base data directory                              |

### Usage Examples

#### Basic Training (Default Parameters)

```bash
python src/train_convlstm_unet.py --map_type FVC
```

#### Custom Parameter Training

```bash
python src/train_convlstm_unet.py \
    --map_type EVI \
    --window_size 10 \
    --patch_size 128 \
    --batch_size 64 \
    --num_epochs 150 \
    --base_lr 0.0005
```

## Output Description

After training, outputs will be generated in the following locations:

### Model Weights

- `models/ConvLSTMUNet_{map_type}_best.pth` - Best model weights

### Visualization Results

- `outputs/plots/ConvLSTMUNet/{map_type}/`
  - `{map_type}_loss_curve.png` - Training loss curve
  - `{map_type}_pred_year{T}.png` - Prediction result map
  - `{map_type}_scatter_year{T}.png` - True vs Predicted scatter plot
  - `{map_type}_loss_history.npz` - Loss history data

## Model Architecture

### ConvLSTM-UNet

The model adopts an encoder-decoder architecture combined with ConvLSTM for temporal modeling:

1. **Encoder**:

   - Two DoubleConv blocks (each containing two 3×3 convolutions + BatchNorm + ReLU)
   - Channel progression: 1 → 32 → 64
   - Uses 2×2 max pooling for downsampling
2. **Bottleneck Layer**:

   - ConvLSTM layer processes temporal information
   - Input/hidden channels: 64
3. **Decoder**:

   - Transposed convolution for upsampling
   - Skip connections preserve spatial details
   - Final output: single-channel continuous values

### Training Strategy

- **Loss Function**: Masked MSE loss (automatically ignores NaN values)
- **Optimizer**: Adam (initial learning rate 0.001)
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.1, patience=5)
- **Early Stopping**: patience=10
- **Mixed Precision Training**: Automatically enabled when GPU is available (AMP)

## Data Format

### Input Data

- **Format**: NPZ files
- **Shape**: `(T, H, W)` - (Time, Height, Width)
- **Data Type**: float32
- **Invalid Values**: NaN

### Mask Files

- **Format**: PKL files
- **Content**: Dictionary containing `mask` and `original_shape`
- **Mask Values**: 0 = valid region, 255 = masked region

## Evaluation Metrics

The model is evaluated using the following regression metrics:

- **MSE** (Mean Squared Error): Mean squared error
- **MAE** (Mean Absolute Error): Mean absolute error
- **RMSE** (Root Mean Squared Error): Root mean squared error
- **R²** (Coefficient of Determination): Coefficient of determination
