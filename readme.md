# IGU-Aug: Information-guided unsupervised augmentation and pixel-wise contrastive learning for medical image analysis

### 1. train CC2D model
```
python -m sc.ssl.ssl --tag debug
```

### 2. Get sift points

```
python sift_select.py
```

### 3. calculate mutual information
```
python test_all_mi.py
```

### 4. Search augmentation params
```
python search_aug.py
```