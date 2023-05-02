_Note: for the reproducibility of the fairness analysis, check out the final section of this readme_

# RHF: ResNet50-HDC-FPN-FasterRCNN-CMFD-LCMFD

[简体中文](https://github.com/shiningxy/RHF/blob/master/README_zh.md) | [English](https://github.com/shiningxy/RHF)

# Paper

[Wang, S., Wang, X., & Guo, X. (2023). Advanced Face Mask Detection Model Using Hybrid Dilation Convolution Based Method. Journal of Software Engineering and Applications, 16(1), 1-19.](https://www.scirp.org/pdf/jsea_2023013111424794.pdf)

Use code, data, weights, etc... Please cite 💝
```
@article{wang2023advanced,
  title={Advanced Face Mask Detection Model Using Hybrid Dilation Convolution Based Method},
  author={Wang, Shaohan and Wang, Xiangyu and Guo, Xin},
  journal={Journal of Software Engineering and Applications},
  volume={16},
  number={1},
  pages={1--19},
  year={2023},
  publisher={Scientific Research Publishing}
}
```

## Model weighting file

* Baidu: https://pan.baidu.com/s/1oW3BopexHkJdsQlb79hsdw  Extraction code: 2swh
* Dropbox: https://www.dropbox.com/s/rg7dqkr71bylaey/save_weights.zip?dl=0
* Google Drive: https://drive.google.com/file/d/1-v7t9nGHauiUbF_o69d0gz52LSry1vSA/view?usp=share_link

## Dataset files CMFD & LCMFD

* Baidu: https://pan.baidu.com/s/1TLEdIfqfQXI-PT49Snv3aQ  Extraction code: 1111


## Environment configuration.
* Python3.6/3.7/3.8
* Pytorch1.7.1
* pycocotools(Linux:``pip install pycocotools``; Windows:``pip install pycocotools-windows``)
* Partial code reference: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

## File structure.
* backbone: feature extraction network ResNet50
* backbone_hdc: ResNet50-HDC The hybrid inflated convolution rate integrated residual network of this paper
* network_files: Faster R-CNN network (including Fast R-CNN and modules such as RPN)
* train_utils: training validation related modules (including cocotools)
* my_dataset.py: custom dataset for reading VOC datasets
* train_mobilenet.py: use MobileNetV2 as the backbone for training
* train_resnet50_fpn.py: use resnet50 + FPN as backbone for training
* train_resnet50_hdc_fpn: train with resnet50-HDC + FPN as backbone
* train_multi_GPU.py: train with multiple GPUs
* predict.py: Simple prediction script to perform prediction tests using trained weights
* validation.py: validate/test the COCO metrics of the data using the trained weights, and generate record_mAP.txt file
* classes.json: pascal_voc tag file


## Pre-trained weights download address (download and put in backbone folder).
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* Note that the downloaded pre-training weights should be renamed, e.g., the ``fasterrcnn_resnet50_fpn_coco.pth`` file is read in train_resnet50_fpn.py
  Not ``fasterrcnn_resnet50_fpn_coco-258fb6c6.pth``
 
 
## dataset

* LCMFD 
  * num of all_annotations: 8232
  * num of with_mask: 3229
  * num of poor_mask: 2813
  * num of none_mask: 2190
        
* CMFD:MaskDatasets_Augment 
  * num of all_annotations: 131422
  * num of with_mask: 53039
  * num of poor_mask: 47203
  * num of none_mask: 31180


## Training method
* Note that the dataset file name in *.py is changed to the dataset file name on your computer
* Modify the parameters in train_res50_fpn.py train_res50_hdc_fpn.py
    * Modify train_res50_fpn.py line 185 to replace data_path
    * Modify train_res50_hdc_fpn.py line 185 Replace data_path
* Make sure the backbone folder contains pre-trained model weights
* If you want to train mobilenetv2+fasterrcnn, use the train_mobilenet.py training script directly
* To train resnet50+fpn+fasterrcnn, use the train_resnet50_fpn.py training script directly
* To train resnet50-hdc+fpn+fasterrcnn, use the train_resnet50_hdc_fpn.py training script directly
* To train with multiple GPUs, use the ``python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py`` command, with the ``nproc_per_node`` parameter being the number of GPUs used
* If you want to specify which GPU devices to use you can prefix the command with ``CUDA_VISIBLE_DEVICES=0,3`` (e.g. I just want to use the 1st and 4th GPU devices in the device)
* ```CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py``

## Caution
* When using the training script, be careful to set '--data-path' (mask_root) to the **root directory** where you store the 'MaskDatasets_Augment' or 'MaskDatasets_NotAugment' folder
* Since Faster RCNN with FPN structure is very memory hungry, if the GPU memory is not enough (if the batch_size is less than 8), it is recommended to use the default norm_layer in the create_model function.
  If the GPU memory is not enough (if the batch_size is less than 8), it is recommended to use the default norm_layer in the create_model function.
* When using the prediction script, set 'train_weights' to your own generated weight path.
* When using the validation file, take care to make sure that your validation set or test set must contain targets for each class, and use it with only '--num-classes', '--data-path' and '--weights', and leave the rest of the code unchanged as much as possible


# Reproducibility of the fairness analysis
Follow these steps:

1. Set up this repo as indicated in the "Environment configuration" section
2. Generate the predicted bounding boxes for a dataset:
   ```python predict.py --images_folder path/to/images --output_folder path/to/predictions```.
   Optionally, you may specify different weights for the model using the `--weights_path` argument.
   The script `predict.py` evaluates all of the images in the folder and produces an output in `.txt` format which includes the file name (without extension), the predicted class, the confidence score of the prediction, and the coordinates of the bounding box in pixels.
3. Produce a `.csv` of the predictions with the normalized coordinates (0 to 1 relative to the image size): ```python adjust_output.py --input path/to/txt_output --output path/to/csv_output --normalize_predictions --dataset_folder path/to/images```. The dataset folder needs to be specify in order to recover the size of the images.
4. Match results with ground truth: ```python match_results_*.py --predictions path/to/csv_predictions --ground_truth path/to/ground_truth --output path/to/output_file```. This will produce a number of csv files, one per protected attribute and metric, containing the rate attained by the model for each termination of the protected attribute, the p-value of the binomial test, and the Cohen's h for the difference.
