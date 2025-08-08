# Two-step Temporal Divisive Clustering for Unsupervised Action Segmentation

## Dataset

50Salads, Breakfast, MPII_Cooking 

Download the data provided by TW-FINCH, which contains theIDT features and the ground truth labels for 3 datasets. 
Downlaod link: https://bwsyncandshare.kit.edu/s/GtWYdcHckJNtRzJ

## Directory Structure

```
root ──  dataset ─── 50Salads/...
      │           ├─ Breakfast/...
      │           └─ MPII_Cooking ─── features/
      │                    ├─ groundTruth/
      │                    └─ mapping/
      │                               └─ mapping.txt
      ├─ TDC ─── read_utils.py
      │           ├─ reduce_classes.py
      │           └─run_on_dataset.py
      │           └─run_TDC.py
      ├─ SSBRN ─── csv/
      │           ├─ libs/
      │           └─imgs/
      │           └─result/
      │           └─scripts/
      │           └─utils/
      │           └─evaluate.py
      │           └─train.py
      ├─ matlab_compute_scores/
      ├ README.md
```


## How to use

First, get clustering labels. We also obtained the accuracy of clustering results through Hungarian matching. If you want more precise results, we saved the clustering labels in the dataset as the req_TDC folder. You can use MATLAB to calculate scores based on these clustering labels.
```
python TDC/run_TDC.py
```

Next, we obtained soft boundary pseudo-labels from the clustering labels.
```
python SS_BRN/utils/generate_boundary_array.py
```
Then, convert ground truth files into numpy array.
```
python SS_BRN/utils/generate_gt_array.py
```

Training SS-BRN
```
python SS-BRN/train.py ./config/xxx/xxx/config.yaml
```
Testing SS-BRN (Get the final results)
```
python SS-BRN/evaluate.py ./config/xxx/xxx/config.yaml
```

If you want to get a more accurate assessment, you can run 
```
matlab/get_scores.m
```
and set "dataset_path" in matlab_compute_scores/get_scores.m
and change "f_path" in matlab_compute_scores/action_seg.m
