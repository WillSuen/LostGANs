## Installation

#### Requiements
* python3
* pytorch 1.0
* numpy
* matplotlib
* opencv

```
pip install -r requiements.txt

# setup for roi_layers
python setup.py build develop
```

#### Data Preparation
Download COCO dataset to `datasets/coco`
```
bash scripts/download_coco.sh
```
Download VG dataset to `datasets/vg`
```
bash scripts/download_vg.sh
python scripts/preprocess_vg.py
```
