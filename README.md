# MI2Former
Efficient Entity Segmentation with Mamba-Driven Lightweight Design

This project offers an implementation of the paper, "[MIFA2Former]". 

<div align="center">
  <img src="main_arch_v21.png" width="90%"/>
</div><br/>

## News
2026-05-10 The dataset, code and pretrained models are released.


## Models
##  Weights of MI2Former.
(1) COCO Entity And EntitySeg Entity
| Alg | Train Data | Model and Config Url | Ap_e |
| ------| ------| ------|------|
| MIFA2Former | COCO Entity | [Google Drive](https://drive.google.com/drive/folders/1wdY1xDXH4JYyBesjAnTD2QsTONPUVEZz?usp=drive_link) |40.3 |
| MIFA2Former | EntitySeg Entity | [Google Drive](https://drive.google.com/drive/folders/1BCghu1TKMew7JaPKxPIJemiktT7WxwTU?usp=drive_link) | 42.7|


(2) ADE20K
| Alg | pre Train Data | Model and Config Url |
| ------| ------| ------|
| MIFA2Former | coco  | [Google Drive](-) |
| MIFA2Former | coco | [Google Drive](-) |


(3) CityScape
| Alg | pre Train Data | Model and Config Url |
| ------| ------| ------|
| MIFA2Former | coco  | [Google Drive](-) |
| MIFA2Former | coco | [Google Drive](-) |

## Data
#### Coco entity dataset
(1) Generate the entity information of each image by the instance and panoptic annotation. Please change the path of coco annotation files in the following code.
```bash
cd /path/to/detectron2/projects/MIFA2Former/make_data
bash make_entity_mask.sh
```
(2) Change the generated entity information to the json files.
```bash
cd /path/to/detectron2/projects/MIFA2Former/make_data
python3 entity_to_json.py
```
### Entityseg entity dataset
(1)  refer to the official repo [EntitySeg-Dataset](https://github.com/adobe-research/EntitySeg-Dataset) for annotation files and image URLs.
For convenience, we provide the images in several links including [Google Drive](https://drive.google.com/drive/folders/1yX2rhOroyhUCGCrmzSm7DL4BfQWvcG0v?usp=drive_link) and [Hugging Face](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg), but we do not own the copyright of the images. It is solely your responsibility to check the original licenses of the images before using them. Any use of the images are at your own discretion and risk. Furthermore, please refer to [the dataset description](DATA.md) on how to set up the dataset before running our code.

## Code
We offer the instructions on installation, evaluation and visualization for the proposed MIFA2Former.

## Installation
This project is based on [Detectron2](https://github.com/facebookresearch/detectron2), which can be constructed as follows.
* Install Detectron2 following [the instructions](https://detectron2.readthedocs.io/tutorials/install.html). We are noting that our code is implemented in detectron2 commit version 28174e932c534f841195f02184dc67b941c65a67 and pytorch 1.8.
* Setup the coco dataset including instance and panoptic annotations following [the structure](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md). The code of entity evaluation metric is saved in the file of modified_cocoapi. You can directly replace your compiled coco.py with modified_cocoapi/PythonAPI/pycocotools/coco.py. 
* Copy this project to `/path/to/detectron2/projects/EntitySeg`
* Set the "find_unused_parameters=True" in distributed training of your own detectron2. You could modify it in detectron2/engine/defaults.py.

## Training
To train model with 8 GPUs, run:
```bash
cd /path/to/detectron2
python3 projects/train_net.py --config-file <projects/MIFA2Former/configs/config.yaml> --num-gpus 8
```

## Evaluation
To evaluate a pre-trained model with 8 GPUs, run:
```bash
cd /path/to/detectron2
python3 projects/MIFA2Former/train_net.py --config-file <config.yaml> --num-gpus 8 --eval-only MODEL.WEIGHTS model_checkpoint
```

## Visualization
To visualize some image result of a pre-trained model, run:
```bash
cd /path/to/detectron2
python3 projects/MIFA2Former/demo_result_and_vis.py --config-file <config.yaml> --input <input_path> --output <output_path> MODEL.WEIGHTS model_checkpoint MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE "True"
```
For example,
```bash
python3 projects/MIFA2Former/demo_result_and_vis.py --config-file projects/MIFA2Former/configs/entity_swin_lw7_1x.yaml --input /data/input/*.jpg --output /data/output MODEL.WEIGHTS /data/pretrained_model/R_50.pth MODEL.CONDINST.MASK_BRANCH.USE_MASK_RESCORE "True"
```

## Params and Flops
<div align="center">
  <img src="ours.png" width="90%"/>
</div><br/>



## <a name="License"></a>License
The code and models are released under the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/).
