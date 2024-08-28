# [Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929) 구현 및 학습
# [Vision Transformer (ViT)](https://arxiv.org/pdf/2010.11929) Implementation and Training
이 프로젝트는 Vision Transformer (ViT)를 구현하고 학습시키는 과정을 포함합니다.  
This project includes the implementation and training process of Vision Transformer (ViT).

ViT는 이미지 분류 작업에서 Transformer 구조를 성공적으로 적용한 모델로, 이 프로젝트에서는 ImageNet-100 데이터셋을 활용하여 ViT 모델을 학습시키고 성능을 평가합니다.  
ViT is a model that successfully applies the Transformer architecture to image classification tasks, and in this project, we train and evaluate the performance of the ViT model using the ImageNet-100 dataset.

## 프로젝트 구성  
## Project Structure

- **Vision Transformer.ipynb**: 이 Jupyter Notebook 파일은 ViT의 전체 구현 및 학습 과정을 포함하고 있습니다.  
- **Vision Transformer.ipynb**: This Jupyter Notebook file contains the complete implementation and training process of ViT.

  데이터 전처리, 모델 구성, 학습 및 평가 과정을 단계별로 설명합니다.  
  It explains the steps of data preprocessing, model configuration, training, and evaluation.

- **AN IMAGE IS WORTH 16X16 WORDS TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (실습).pdf**: 이 PDF 파일은 ViT의 실습 과정과 결과를 제공합니다.  
- **AN IMAGE IS WORTH 16X16 WORDS TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (실습).pdf**: This PDF file provides the practical process and results of ViT.

  ViT의 아키텍처와 코드, 그리고 학습 결과에 대한 설명을 포함합니다.  
  It includes explanations of the ViT architecture, code, and training results.

- **AN IMAGE IS WORTH 16X16 WORDS TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (리뷰).pdf**: 이 PDF 파일은 ViT에 대한 리뷰와 이론적인 배경을 제공합니다.  
- **AN IMAGE IS WORTH 16X16 WORDS TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (리뷰).pdf**: This PDF file provides a review and theoretical background of ViT.

  ViT의 구조, 관련 연구, 방법론, 데이터셋, 모델 변형 및 학습 세부 사항에 대한 설명이 포함되어 있습니다.  
  It includes explanations of the ViT structure, related research, methodology, datasets, model variants, and training details.

## 파일 설명  
## File Descriptions

### 1. Vision Transformer.ipynb

- **데이터 전처리**: ImageNet-100 데이터셋을 전처리하여 학습에 적합한 형태로 변환합니다.  
- **Data Preprocessing**: Preprocess the ImageNet-100 dataset and transform it into a format suitable for training.

  데이터 증강 기법을 사용하여 모델의 일반화 성능을 향상시킵니다.  
  Use data augmentation techniques to improve the model's generalization performance.

- **모델 구현**: ViT의 주요 레이어(패치 생성, 패치 인코더, MHA 등)를 구현하고, 모델을 구성합니다.  
- **Model Implementation**: Implement the key layers of ViT (patch creation, patch encoder, MHA, etc.), and build the model.

- **모델 학습**: Adam 옵티마이저를 사용하여 모델을 학습시키며, 학습 중에 ModelCheckpoint와 EarlyStopping을 사용하여 최적의 모델을 저장하고 학습을 조기 종료합니다.  
- **Model Training**: Train the model using the Adam optimizer, and during training, save the best model and stop early using ModelCheckpoint and EarlyStopping.

- **모델 평가**: 학습된 모델을 사용하여 검증 데이터셋에서의 성능을 평가하고, 최종적으로 모델의 정확도와 손실을 시각화합니다.  
- **Model Evaluation**: Evaluate the performance of the trained model on the validation dataset and visualize the model's accuracy and loss.

### 2. AN IMAGE IS WORTH 16X16 WORDS TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (실습).pdf

- **ViT 아키텍처**: ViT의 전체 구조를 그림으로 설명하며, 각 레이어의 역할과 연결 관계를 이해할 수 있습니다.  
- **ViT Architecture**: Illustrates the overall structure of ViT, helping to understand the role and connections of each layer.

- **코드 설명**: ViT의 각 코드 부분에 대한 설명을 포함하고, create_patches, patchEncoder, visionTransformer 등 주요 함수와 클래스의 동작을 시각적으로 보여줍니다.  
- **Code Explanation**: Includes explanations for each part of the ViT code and visually shows the operation of key functions and classes such as create_patches, patchEncoder, and visionTransformer.

- **학습 결과**: CIFAR-10과 ImageNet-100 데이터셋에서의 학습 결과를 보여주며, 손실과 정확도 변화를 시각적으로 표현합니다.  
- **Training Results**: Shows the training results on CIFAR-10 and ImageNet-100 datasets and visually represents the changes in loss and accuracy.

### 3. AN IMAGE IS WORTH 16X16 WORDS TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE (리뷰).pdf

- **ViT 소개**: ViT의 이론적 배경과 이미지 분류 작업에서의 중요성에 대해 설명합니다.  
- **ViT Introduction**: Explains the theoretical background of ViT and its significance in image classification tasks.

- **관련 연구**: Self-Attention과 Transformer 구조를 이미지에 적용한 기존 연구들을 소개합니다.  
- **Related Work**: Introduces previous studies that applied Self-Attention and Transformer structures to images.

- **방법론**: ViT의 학습 방법론과 하이브리드 아키텍처, 그리고 모델 변형에 대해 설명합니다.  
- **Methodology**: Explains the training methodology of ViT, the hybrid architecture, and the model variants.

- **결론**: ViT의 주요 기여와 한계를 정리하고, 이미지 인식 분야에서의 미래 가능성을 논의합니다.  
- **Conclusion**: Summarizes the key contributions and limitations of ViT and discusses future possibilities in the field of image recognition.

## 사용 방법  
## Usage

1. **환경 설정**: 프로젝트를 실행하기 위해 필요한 라이브러리와 패키지를 설치합니다.  
1. **Setup Environment**: Install the necessary libraries and packages to run the project.

   주로 사용되는 라이브러리는 `tensorflow`, `keras`, `numpy`, `matplotlib` 등이 있습니다.  
   Commonly used libraries include `tensorflow`, `keras`, `numpy`, `matplotlib`, etc.

2. **데이터 준비**: ImageNet-100 데이터셋을 준비하여 적절한 디렉토리에 저장합니다.  
2. **Prepare Data**: Prepare the ImageNet-100 dataset and store it in the appropriate directory.

   `Vision Transformer.ipynb` 파일 내에서 데이터 경로를 지정하여 사용합니다.  
   Specify the data path within the `Vision Transformer.ipynb` file.

3. **모델 학습**: `Vision Transformer.ipynb`를 실행하여 모델을 학습시킵니다.  
3. **Model Training**: Run `Vision Transformer.ipynb` to train the model.

   학습 중간에 모델이 자동으로 저장되며, EarlyStopping을 사용하여 과적합을 방지합니다.  
   The model is automatically saved during training, and EarlyStopping is used to prevent overfitting.

4. **모델 평가**: 학습된 모델을 사용하여 검증 데이터셋에서 모델의 성능을 평가하고, 결과를 시각화합니다.  
4. **Model Evaluation**: Evaluate the trained model using the validation dataset and visualize the results.

## 주의 사항  
## Notes

- ImageNet-100 데이터셋은 매우 크기 때문에 학습 시 GPU 사용을 권장합니다.  
- The ImageNet-100 dataset is very large, so it is recommended to use a GPU for training.

- 학습 과정에서 충분한 메모리와 저장 공간을 확보해야 합니다.  
- Ensure that you have sufficient memory and storage space during the training process.

## 기여  
## Contribution

이 프로젝트에 기여하고 싶다면, Pull Request를 제출해 주세요.  
If you would like to contribute to this project, please submit a Pull Request.

버그 리포트나 개선 사항 제안도 환영합니다.  
Bug reports and suggestions for improvements are also welcome.
