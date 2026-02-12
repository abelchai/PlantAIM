## Table of Contents
- [Abstract](#abstract)
- [Contribution](#contribution)
- [Proposed Model](#proposed-model)
- [Results](#result)
- [Preparation](#dataset-and-pretrained-weight)
- [Implementation](#implementations)
- [License](#license)
- [Citation](#citation)

# PlantAIM: A New Baseline Model Integrating Global Attention and Local Features for Enhanced Plant Disease Identification

## Abstract
This paper is accepted on Smart Agricultural Technology 2025 [[Link]](https://www.sciencedirect.com/journal/smart-agricultural-technology)

[[Paper]](https://www.sciencedirect.com/science/article/pii/S2772375525000474)

Plant diseases significantly affect the quality and yield of agricultural production. Conventionally, detection has relied on plant pathologists, but recent advances in deep learning, particularly the Vision Transformer (ViT) and Convolutional Neural Network (CNN), have made it feasible for automated plant disease identification. Despite their prominence, there are still significant gaps in our understanding of how these models differ in feature extraction and representation, particularly in complex multi-crop disease identification tasks. This challenge arises from the simultaneous need to learn crop-specific and disease-specific features for accurate identification of crop species and its associated diseases. To address this, we introduce Plant Disease Glocal-Local Features Fusion Attention Model (PlantAIM), a new hybrid framework that fuses global attention mechanisms of ViT with local feature extraction capabilities of CNN. PlantAIM aims to improve the model's ability to simultaneously learn and focus on crop-specific and disease-specific features. We conduct extensive evaluations to assess the robustness and generalizability of PlantAIM compared to state-of-the-art (SOTA) models, including scenarios with limited training samples and real-world environmental data. Our results show that PlantAIM achieves superior performance. This research not only deepens our understanding of feature learning for ViT and CNN models, but also sets a new benchmark in the dynamic field of plant disease identification. The code will be made available upon publication.

<p align="center">
  <img src="figure/intro.png" alt="intro" width="800" height="600">
</p>

## Contribution
1. We introduce novel Plant Disease Global-Local Features Fusion Attention model (PlantAIM), which combines ViT and CNN components to enhance feature extraction for multi-crop plant disease identification.
2. Our experimental results demonstrate PlantAIM's exceptional robustness and generalization, achieving state-of-the-art performance in both controlled environments and real-world scenarios.
3. Our feature visualization analysis reveals that CNNs emphasize plant patterns, while ViTs focus on disease symptoms.

## Proposed model
Plant Disease Global-Local Features Fusion Attention model (PlantAIM) [[code]](model/)
  * Key feature: combines ViT and CNN components to enhance feature extraction for multi-crop plant disease identification.

<p align="center">
  <img src="figure/PlantAIM.png" alt="CL-ViT" width="800">
  <br>
  <i>Proposed PlantAIM architecture.</i>
</p>

## Result
<p align="center">
  <img src="figure/results.png" alt="results" width="700">
</p>

## Grad-CAM visualization result
![tomato Results](figure/tomato.png)
![cherry Results](figure/cherry.png)
![apple Results](figure/apple.png)

## Dataset and pretrained weight
* PV Dataset: [spMohanty Github](https://github.com/spMohanty/PlantVillage-Dataset/tree/master)  
(You can group all images into single folder to directly use the csv file provided in this repo)
* PlantDoc dataset: [Kaggle](https://www.kaggle.com/datasets/abdulhasibuddin/plant-doc-dataset) 

* IPM and Bing dataset will be release soon

* download [ViT pretrained weight](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth) link (From [rwightman Github timm repo](https://github.com/huggingface/pytorch-image-models))

## Implementations
PlantAIM (2H) >> [pytorch implementation code](model/PlantAIM_2H.py)

PlantAIM (1H) >> [pytorch implementation code](model/PlantAIM_1H.py)

Notes
* The csv file (metadata of images) are [here](dataset/) 

## Virtual environment dependencies
Python 3.12.9
```
python -m venv py
cd .\py\Scripts
activate
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements_plantaim.txt
```

## License

Creative Commons Attribution-Noncommercial-NoDerivative Works 4.0 International License (“the [CC BY-NC-ND License](https://creativecommons.org/licenses/by-nc-nd/4.0/)”)

## See also

1. Pairwise Feature Learning for Unseen Plant Disease Recognition [[Paper](https://ieeexplore.ieee.org/abstract/document/10222401/)] 

   The first implementation of FF-ViT model with moving weighted sum. The current work improved and evaluated the performance of FF-ViT model on larger-scale dataset.
  
3. Unveiling Robust Feature Spaces: Image vs. Embedding-Oriented Approaches for Plant Disease Identification [[Paper](https://ieeexplore.ieee.org/abstract/document/10317550/)]

   The analysis between image or embedding feature space for plant disease identifications.

4. PlantAIM: A New Baseline Model Integrating Global Attention and Local Features for Enhanced Plant Disease Identification [[Paper](https://www.sciencedirect.com/science/article/pii/S2772375525000474)] [[Github](https://github.com/abelchai/PlantAIM)]
  
   Plant Disease Global-Local Features Fusion Attention model (PlantAIM) model which combines ViT and CNN components to enhance feature extraction for multi-crop plant disease identification.
   
6. Beyond-supervision-Harnessing-self-supervised-learning-in-unseen-plant-disease-recognition [[Paper](https://www.sciencedirect.com/science/article/pii/S0925231224013791)] [[Github](https://github.com/abelchai/Cross-Learning-Vision-Transformer-CL-ViT)]
 
   Cross Learning Vision Transformer (CL-ViT) model that incorporating self-supervised learning into a supervised model.

7. Can Language Improve Visual Features For Distinguishing Unseen Plant Diseases? [[Paper](https://link.springer.com/chapter/10.1007/978-3-031-78113-1_20)] [[Github](https://github.com/abelchai/FF-CLIP-Can-Language-Improve-Visual-Features-For-Distinguishing-Unseen-Plant-Diseases)]

   FF-CLIP model that incorporate textual data as language cues in guiding visual features to improve the identification of unseen plant diseases.

8. Deep-Plant-Disease Dataset Is All You Need for Plant Disease Identification [[Paper](https://dl.acm.org/doi/abs/10.1145/3746027.3758192)] [[Github](https://github.com/abelchai/Deep-Plant-Disease-Dataset-Is-All-You-Need-for-Plant-Disease-Identification)]

   We curated the largest plant disease dataset with text descriptions known as Deep-Plant-Disease, comprising 248,578 images across 55 crop species, 175 disease classes, and 333 unique crop-disease compositions. We also conducted comprehensive benchmarking across   multiple downstream tasks in plant disease identification under diverse conditions that simulate different real-world challenges.
   
## Citation

```bibtex
@article{chai2025plantaim,
  title={PlantAIM: A New Baseline Model Integrating Global Attention and Local Features for Enhanced Plant Disease Identification},
  author={Chai, Abel Yu Hao and Lee, Sue Han and Tay, Fei Siang and Go{\"e}au, Herv{\'e} and Bonnet, Pierre and Joly, Alexis},
  journal={Smart Agricultural Technology},
  pages={100813},
  year={2025},
  publisher={Elsevier}
}
