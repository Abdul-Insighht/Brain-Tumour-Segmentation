# 3D Brain Tumor Segmentation System

![Brain Tumor Segmentation]

A deep learning-based web application for automatic segmentation of brain tumors from 3D MRI scans using an attention-based U-Net architecture. This application provides an intuitive web interface for uploading MRI data and visualizing segmentation results in both 3D and 2D views.

## Features

### Core Functionality
- **Automatic Brain Tumor Segmentation**: Uses a trained Attention U-Net model to segment brain tumors from MRI scans
- **Multi-Modal MRI Support**: Processes FLAIR, T1CE, and T2 MRI sequences
- **3D Visualization**: Interactive 3D visualization of brain structures and tumor regions using VTK/PyVista
- **2D Slice Visualization**: View MRI data and segmentation masks across multiple slices (X, Y, Z axes)
- **Real-time Inference**: On-the-fly segmentation processing with configurable parameters

### Interactive Controls
- **Modality Selection**: Toggle visibility of different MRI sequences (FLAIR, T1CE, T2)
- **Opacity Adjustment**: Control transparency of brain structures and segmentation masks
- **Threshold Controls**: Adjust visualization thresholds for better rendering
- **Mask Categories**: Visualize different tumor regions (NEC, ED, ET) with color coding
- **Camera Controls**: Reset camera view and interact with 3D models

## Architecture

### Deep Learning Model
- **Model Type**: Attention U-Net (3D)
- **Input Shape**: (192, 192, 3) - 2D slices with 3 MRI modalities
- **Classes**: 4 (Background, Non-Enhancing Tumor, Edema, Enhancing Tumor)
- **Loss Function**: Combined Dice Loss + Focal Loss
- **Preprocessing**: CLAHE (Contrast Limited Adaptive Histogram Equalization)

### Technology Stack
- **Frontend**: Streamlit
- **3D Visualization**: VTK, PyVista, stpyvista
- **Deep Learning**: TensorFlow/Keras
- **Medical Imaging**: Nibabel (NIfTI format support)
- **Image Processing**: OpenCV, NumPy, Matplotlib

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for faster inference)

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd Brain_Tumour_Seg
```

2. **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

### Complete Pipeline (main.py)

The `main.py` script orchestrates the complete pipeline: preprocessing, training, and launching the GUI.

#### Full Pipeline (preprocess, train, and run GUI)
```bash
python main.py --data-dir /path/to/MICCAI_BraTS2020_TrainingData
```

#### Train on Preprocessed Data
```bash
python main.py --skip-preprocessing --numpy-dir /path/to/numpy_data
```

#### Only Run GUI with Existing Model
```bash
python main.py --only-gui --model-path /path/to/model.h5
```

#### Available Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--data-dir` | Path to BraTS2020 training data directory | Required* |
| `--numpy-dir` | Path to store/read preprocessed numpy files | `numpy_data` |
| `--output-dir` | Path to save trained model outputs | `trained_model` |
| `--batch-size` | Batch size for training | `5` |
| `--epochs` | Number of training epochs | `80` |
| `--learning-rate` | Learning rate | `1e-4` |
| `--skip-preprocessing` | Skip preprocessing step | - |
| `--skip-training` | Skip training step | - |
| `--skip-gui` | Skip GUI launch | - |
| `--only-gui` | Only run the GUI | - |
| `--model-path` | Path to trained model file | - |
| `--show-plots` | Show plots during preprocessing | - |

*Required unless using `--skip-preprocessing` or `--only-gui`

### Individual Scripts

#### Data Preprocessing
```bash
python DataPreprocessing.py --data-dir /path/to/MICCAI_BraTS2020_TrainingData --output-dir numpy_data
```

#### Model Training
```bash
python Attention_Model_Clahe1024.py --data-dir numpy_data --output-dir trained_model --epochs 80 --batch-size 5
```

#### Run GUI Application
```bash
streamlit run app.py -- --model-path /path/to/model.h5
```

### GUI Usage Steps

1. **Upload MRI Scans**: Upload three MRI sequences in NIfTI format (FLAIR, T1CE, T2)
2. **Configure Visualization**: Adjust parameters in the sidebar (thresholds, opacity)
3. **Select Modalities**: Choose which MRI sequences to display
4. **Explore Tumor Regions**: Visualize different tumor categories with color coding
5. **3D Interaction**: Rotate (click+drag), Zoom (scroll), Pan (Shift+click+drag)

## Dataset

This project uses the **BraTS (Brain Tumor Segmentation) 2020 Dataset**:

- **Training Data**: 369 multi-modal MRI scans (FLAIR, T1, T1CE, T2)
- **Data Format**: NIfTI (.nii) files
- **Image Dimensions**: 240 x 240 x 155 voxels
- **Preprocessing**:
  - Cropped to brain region (192 x 192 x 128)
  - Normalized using Min-Max scaling
  - CLAHE enhancement applied
  - Segmentation masks: 4 classes

### Tumor Classes
- **Class 0**: Background (Healthy tissue)
- **Class 1**: Non-Enhancing Tumor (NEC)
- **Class 2**: Edema (ED)
- **Class 3**: Enhancing Tumor (ET)

## Project Structure

```
Brain_Tumour_Seg/
├── main.py                          # Main pipeline script
├── app.py                           # Streamlit GUI application
├── Attention_Model_Clahe1024.py     # Model training script
├── DataPreprocessing.py             # Data preprocessing pipeline
├── DoubleAttention1024.py           # Attention U-Net architecture
├── image_loader.py                  # Image loading utilities
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # Project documentation
```

## Technical Details

### Model Architecture
The Attention U-Net incorporates:
- **Encoder-Decoder Structure**: Contracting and expansive paths
- **Skip Connections**: Preserve fine-grained details
- **Attention Gates**: Focus on relevant features
- **Multi-Scale Feature Extraction**: Capture tumors at various scales

### Performance Metrics
The model is evaluated using:
- **Dice Score**: Measures overlap between predicted and ground truth
- **IoU (Intersection over Union)**: Spatial accuracy metric
- **Precision & Recall**: Classification performance
- **F1-Score**: Harmonic mean of precision and recall

### Data Preprocessing Pipeline
1. **Loading**: Load NIfTI files using Nibabel
2. **Normalization**: Min-Max scaling to [0, 1] range
3. **Cropping**: Extract brain region of interest
4. **CLAHE**: Enhance contrast for better feature visibility
5. **Stacking**: Combine 3 modalities into 4D array
6. **Model Input**: 192x192x3 patches for inference

## Color Coding

The application uses the following color scheme for segmentation visualization:

- **Red**: Non-Enhancing Tumor (NEC)
- **Blue**: Edema (ED)
- **Green**: Enhancing Tumor (ET)
- **Yellow**: Additional annotations (if present)
- **White**: Brain tissue (MRI modalities)

## Configuration Options

### Model Parameters
```python
INPUT_SHAPE = (192, 192, 3)
NUM_CLASSES = 4
START, END = 34, 226
Z_START, Z_END = 13, 141
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILE_GRID_SIZE = (9, 9)
```

### Visualization Parameters
```python
BRAIN_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5
BRAIN_OPACITY = 0.4
MASK_OPACITY = 0.9
```
## Acknowledgments

- **BraTS Community**: For providing the benchmark dataset
- **Medical Imaging Community**: For foundational research in brain tumor segmentation
- **TensorFlow/Keras**: For the deep learning framework
- **PyVista/VTK**: For 3D visualization capabilities

## References

1. **BraTS 2020 Dataset**: [MICCAI BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/)
2. **Attention U-Net**: "Attention U-Net: Learning Where to Look for the Pancreas"
3. **Segmentation Models 3D**: GitHub repository for 3D segmentation architectures

---

**Note**: This application is intended for research and educational purposes only. It should not be used for clinical diagnosis without proper validation and regulatory approval.


---

## 📬 Contact

**Hafiz Abdul Rehman**

- 📧 Email: hafizrehman3321@gmail.com
- 💼 LinkedIn: [Hafiz Abdul Rehman](https://linkedin.com/in/hafiz-abdul-rehman-9990ab329)
- 🐙 GitHub: [Abdul-Insighht](https://github.com/Abdul-Insighht)

---

## 🌟 Show Your Support

If you find this project helpful, please consider:

- ⭐ **Starring** this repository
- 🔄 **Sharing** with others
- 🐛 **Reporting** issues
- 💡 **Suggesting** improvements

---

<p align="center">Made with ❤️ by <b>Hafiz Abdul Rehman</b></p>
