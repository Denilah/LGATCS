# LGATCS
A Local-to-Global Attention Transformer for Code Search.

## Dependency
Tested in Ubuntu 18.04.6
Python 3.6.0
torch 1.10.1
numpy 1.24.4
tqdm 4.65.0

## Usage
### Dataset
We use the dataset shared by @guxd. You can download this shared dataset from [Google Drive](https://drive.google.com/drive/folders/1GZYLT_lzhlVczXjD6dgwVUvDDPHMB6L7?usp=sharing) and add this dataset folder to `/data`. Meanwhile, we offer a sample dataset `/data/github` for you to test. 

### Configuration
Edit hyper-parameters and settings in `config.py`

### train
```bash
python train.py --mode train
```

### eval
```bash
python train.py --mode eval
```
Or you can use our trained model `/model_save/joint_embed_model_test.h5` (on the sample data set) to test directly.