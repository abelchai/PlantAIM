# PlantAIM: A New Baseline Model Integrating Global Attention and Local Features for Enhanced Plant Disease Identification

![PlantAIM](figure/PlantAIM.png)
<p align="center">Proposed PlantAIM architecture.</p>

The contributions of this paper:
1. We introduce novel Plant Disease Global-Local Features Fusion Attention model (PlantAIM), which combines ViT and CNN components to enhance feature extraction for multi-crop plant disease identification.
2. Our experimental results demonstrate PlantAIM's exceptional robustness and generalization, achieving state-of-the-art performance in both controlled environments and real-world scenarios.
3. Our feature visualization analysis reveals that CNNs emphasize plant patterns, while ViTs focus on disease symptoms. By leveraging these characteristics, PlantAIM sets a new benchmark in multi-crop plant disease identification.

## Acc Result
![Acc Results](result/result.png)

## Grad-CAM visualization result
![tomato Results](result/tomato.png)
![cherry Results](result/cherry.png)
![apple Results](result/apple.png)

## Preparation
* Dataset: [spMohanty Github](https://github.com/spMohanty/PlantVillage-Dataset/tree/master)  
(You can group all images into single folder to directly use the csv file provided in this repo)

* download [ViT pretrained weight](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) link (From [rwightman Github timm repo](https://github.com/huggingface/pytorch-image-models))

## Implementations
PlantAIM (2H) >> [code](model/CL-ViT.py)

PlantAIM (1H) >> [code](model/CL-ViT.py)

Notes
* The csv file (metadata of images) are [here](dataset/csv_FFViT/) 
