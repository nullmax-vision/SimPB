## Prepare Dataset
### NuScenes
**1. Download [nuScenes](https://www.nuscenes.org/download) V1.0 dataset**

**2. Link dataset to project**
```bash
ln -s path/to/nuscenes ./data/nuscenes
```
**3. Convert nuscenes dataset**
```bash
python tools/data_converter/nuscenes_converter.py --info_prefix ./data/nuscenes/simpb_nuscenes
```
### Kmean Anchors
```bash
python tools/anchor_generator.py --ann_file ./data/nuscenes/simpb_nuscenes_infos_train.pkl
```

**Folder structure**
```
SimPB
├── projects/
├── tools/
├── ckpts/
├── data/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_kmeans900.npy
|   |   ├── simpb_nuscenes_infos_test.pkl
|   |   ├── simpb_nuscenes_infos_train.pkl
|   |   ├── simpb_nuscenes_infos_val.pkl
```