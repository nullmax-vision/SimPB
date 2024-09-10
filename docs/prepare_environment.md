## Prepare Environment
* Linux
* python 3.8
* Pytorch 1.10.0+
* CUDA 11.1+

**1. Create a conda virtual environment**
```bash
conda create -n simpb python=3.8 -y
conda activate simpb
```

**2. Install Pytorch**
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

**3. Install other packages**
```bash
pip install --upgrade pip
pip install -r requirement.txt
```

**4. Compile the deformable_aggregation CUDA op**
```bash
cd projects/mmdet3d_plugin/ops
python setup.py develop
cd ../../../
```