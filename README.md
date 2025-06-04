# P3P: Pseudo-3D Pre-training for Scaling 3D Voxel-based Masked Autoencoders [[arXiv]](https://arxiv.org/pdf/2408.10007)

## TODO List
- [x] Release all checkpoints.
- [x] Release pre-training and fine-tuning code.
- [x] Release data.
- [ ] Write instructions.

## Pre-training Dataset
Download "train_depth_v2.zip" at our hugging face [website](https://huggingface.co/datasets/XuechaoChen/P3P-Lift).
Download training set of ImageNet-1K at their [official website](https://www.image-net.org/download.php).

Dataset organization:
```
│YourDataPath/
├──train/
│   ├──123456xxx.JPEG/
│   ├──.......
├──train_depth_v2/
│   ├──123456xxx._img_depth.npy/
│   ├──.......
```

## Fine-tuning Datasets
See hugging face 

## Pre-trained Model
See hugging face (https://huggingface.co/XuechaoChen/P3P-MAE)
