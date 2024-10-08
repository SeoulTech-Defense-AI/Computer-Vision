# [CMT (Convolutional Neural Networks Meet Vision Transformers)](https://arxiv.org/pdf/2107.06263) 구현 및 학습
# [CMT (Convolutional Neural Networks Meet Vision Transformers)](https://arxiv.org/pdf/2107.06263) Implementation and Training

이 프로젝트는 CMT (Convolutional Neural Networks Meet Vision Transformers)를 구현하고 학습시키는 과정을 포함합니다.  
This project includes the implementation and training process of CMT (Convolutional Neural Networks Meet Vision Transformers).

CMT는 CNN과 Vision Transformer의 장점을 결합한 모델로, 이 프로젝트에서는 CIFAR-10 및 ImageNet-100 데이터셋을 활용하여 CMT 모델을 학습시키고 성능을 평가합니다.  
CMT is a model that combines the strengths of CNN and Vision Transformer. In this project, we train and evaluate the performance of the CMT model using the CIFAR-10 and ImageNet-100 datasets.

## 프로젝트 구성  
## Project Structure

- **CMT.ipynb**: 이 Jupyter Notebook 파일은 CMT의 전체 구현 및 학습 과정을 포함하고 있습니다.  
- **CMT.ipynb**: This Jupyter Notebook file contains the complete implementation and training process of CMT.

  데이터 전처리, 모델 구성, 학습 및 평가 과정을 단계별로 설명합니다.  
  It explains the steps of data preprocessing, model configuration, training, and evaluation.

- **CMT_Convolutional Neural Networks Meet Vision Transformers (실습).pdf**: 이 PDF 파일은 CMT의 실습 과정과 결과를 제공합니다.  
- **CMT_Convolutional Neural Networks Meet Vision Transformers (실습).pdf**: This PDF file provides the practical process and results of CMT.

  CMT의 아키텍처, 코드, 그리고 학습 결과에 대한 설명을 포함합니다.  
  It includes explanations of the CMT architecture, code, and training results.

- **CMT_Convolutional Neural Networks Meet Vision Transformers (리뷰).pdf**: 이 PDF 파일은 CMT에 대한 리뷰와 이론적인 배경을 제공합니다.  
- **CMT_Convolutional Neural Networks Meet Vision Transformers (리뷰).pdf**: This PDF file provides a review and theoretical background of CMT.

  CMT의 구조, 관련 연구, 방법론, 데이터셋, 모델 변형 및 학습 세부 사항에 대한 설명이 포함되어 있습니다.  
  It includes explanations of the CMT structure, related research, methodology, datasets, model variants, and training details.

## 파일 설명  
## File Descriptions

### 1. CMT.ipynb

- **데이터 전처리**: CIFAR-10 및 ImageNet-100 데이터셋을 전처리하여 학습에 적합한 형태로 변환합니다.  
- **Data Preprocessing**: Preprocess the CIFAR-10 and ImageNet-100 datasets and transform them into a format suitable for training.

  데이터 증강 기법을 사용하여 모델의 일반화 성능을 향상시킵니다.  
  Use data augmentation techniques to improve the model's generalization performance.

- **모델 구현**: CMT의 주요 레이어(LPU, LMHSA, IRFFN 등)를 구현하고, 모델을 구성합니다.  
- **Model Implementation**: Implement the key layers of CMT (LPU, LMHSA, IRFFN, etc.), and build the model.

- **모델 학습**: Adam 옵티마이저를 사용하여 모델을 학습시키며, 학습 중에 ModelCheckpoint와 EarlyStopping을 사용하여 최적의 모델을 저장하고 학습을 조기 종료합니다.  
- **Model Training**: Train the model using the Adam optimizer, and during training, save the best model and stop early using ModelCheckpoint and EarlyStopping.

- **모델 평가**: 학습된 모델을 사용하여 검증 데이터셋에서의 성능을 평가하고, 최종적으로 모델의 정확도와 손실을 시각화합니다.  
- **Model Evaluation**: Evaluate the performance of the trained model on the validation dataset and visualize the model's accuracy and loss.

### 2. CMT_Convolutional Neural Networks Meet Vision Transformers (실습).pdf

- **CMT 아키텍처**: CMT의 전체 구조를 그림으로 설명하며, 각 레이어의 역할과 연결 관계를 이해할 수 있습니다.  
- **CMT Architecture**: Illustrates the overall structure of CMT, helping to understand the role and connections of each layer.

- **코드 설명**: CMT의 각 코드 부분에 대한 설명을 포함하고, LPU, LMHSA, IRFFN 등 주요 함수와 클래스의 동작을 시각적으로 보여줍니다.  
- **Code Explanation**: Includes explanations for each part of the CMT code and visually shows the operation of key functions and classes such as LPU, LMHSA, and IRFFN.

- **학습 결과**: CIFAR-10과 ImageNet-100 데이터셋에서의 학습 결과를 보여주며, 손실과 정확도 변화를 시각적으로 표현합니다.  
- **Training Results**: Shows the training results on CIFAR-10 and ImageNet-100 datasets and visually represents the changes in loss and accuracy.

### 3. CMT_Convolutional Neural Networks Meet Vision Transformers (리뷰).pdf

- **CMT 소개**: CMT의 이론적 배경과 이미지 분류 작업에서의 중요성에 대해 설명합니다.  
- **CMT Introduction**: Explains the theoretical background of CMT and its significance in image classification tasks.

- **관련 연구**: CNN과 Transformer 구조를 이미지에 적용한 기존 연구들을 소개합니다.  
- **Related Work**: Introduces previous studies that applied CNN and Transformer structures to images.

- **방법론**: CMT의 학습 방법론과 하이브리드 아키텍처, 그리고 모델 변형에 대해 설명합니다.  
- **Methodology**: Explains the training methodology of CMT, the hybrid architecture, and the model variants.

- **결론**: CMT의 주요 기여와 한계를 정리하고, 이미지 인식 분야에서의 미래 가능성을 논의합니다.  
- **Conclusion**: Summarizes the key contributions and limitations of CMT and discusses future possibilities in the field of image recognition.

## 사용 방법  
## Usage

1. **환경 설정**: 프로젝트를 실행하기 위해 필요한 라이브러리와 패키지를 설치합니다.  
1. **Setup Environment**: Install the necessary libraries and packages to run the project.

   주로 사용되는 라이브러리는 `tensorflow`, `keras`, `numpy`, `matplotlib` 등이 있습니다.  
   Commonly used libraries include `tensorflow`, `keras`, `numpy`, `matplotlib`, etc.

2. **데이터 준비**: CIFAR-10 및 ImageNet-100 데이터셋을 준비하여 적절한 디렉토리에 저장합니다.  
2. **Prepare Data**: Prepare the CIFAR-10 and ImageNet-100 datasets and store them in the appropriate directory.

   `CMT.ipynb` 파일 내에서 데이터 경로를 지정하여 사용합니다.  
   Specify the data path within the `CMT.ipynb` file.

3. **모델 학습**: `CMT.ipynb`를 실행하여 모델을 학습시킵니다.  
3. **Model Training**: Run `CMT.ipynb` to train the model.

   학습 중간에 모델이 자동으로 저장되며, EarlyStopping을 사용하여 과적합을 방지합니다.  
   The model is automatically saved during training, and EarlyStopping is used to prevent overfitting.

4. **모델 평가**: 학습된 모델을 사용하여 검증 데이터셋에서 모델의 성능을 평가하고, 결과를 시각화합니다.  
4. **Model Evaluation**: Evaluate the trained model using the validation dataset and visualize the results.

## 주의 사항  
## Notes

- CIFAR-10 및 ImageNet-100 데이터셋은 매우 크기 때문에 학습 시 GPU 사용을 권장합니다.  
- The CIFAR-10 and ImageNet-100 datasets are very large, so it is recommended to use a GPU for training.

- 학습 과정에서 충분한 메모리와 저장 공간을 확보해야 합니다.  
- Ensure that you have sufficient memory and storage space during the training process.

## 기여  
## Contribution

이 프로젝트에 기여하고 싶다면, Pull Request를 제출해 주세요.  
If you would like to contribute to this project, please submit a Pull Request.

버그 리포트나 개선 사항 제안도 환영합니다.  
Bug reports and suggestions for improvements are also welcome.