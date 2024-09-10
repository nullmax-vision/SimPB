## Getting Started

## Train

**1. Download pretrained backbone**
```bash
cd ckpts
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
```

**2. Train simpb with multiple GPUs**
```bash
bash ./tools/dist_train.sh ./projects/configs/simpb_nus_r50_img_704x256.py 8 --no-validate
```

## Test
**1. Download pretrained model**

download pretrained model [here](https://github.com/nullmax-vision/SimPB/releases/download/untagged-57f40bcb241e19ede053/simpb_r50_img.pth), or use your own training weight

**2. Evaluate the pretrained model**
```bash
bash ./tools/dist_test.sh ./projects/configs/simpb_nus_r50_img_704x256.py path/to/model.pth 8 --eval bbox
```

## Visualize
**1. Get results file**
```bash
python ./tools/test.py ./projects/configs/simpb_nus_r50_img_704x256.py path/to/model.pth --out path/to/model.pkl
```

**2. Load and show results**
```bash
python ./tools/test.py ./projects/configs/simpb_nus_r50_img_704x256.py path/to/model.pth --result_file path/to/model.pkl --show_only --show-dir ./
```