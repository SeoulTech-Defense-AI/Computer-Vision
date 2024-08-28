# 커스텀 모델 구현 및 실험
# Custom Model Implementations and Experiments

이 저장소는 [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), [VGG](https://arxiv.org/pdf/1409.1556), [ResNet](https://arxiv.org/pdf/1512.03385), [Xception](https://arxiv.org/pdf/1610.02357)과 같은 인기 있는 아키텍처를 기반으로 한 다양한 커스텀 딥러닝 모델을 포함하고 있습니다. 이러한 모델들은 ImageNet 데이터셋을 사용한 이미지 분류 작업을 위해 설계, 학습, 평가되었습니다.
This repository contains various custom deep learning models built on popular architectures such as [AlexNet](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf), [VGG](https://arxiv.org/pdf/1409.1556), [ResNet](https://arxiv.org/pdf/1512.03385), and [Xception](https://arxiv.org/pdf/1610.02357). The models were designed, trained, and evaluated for image classification tasks using the ImageNet dataset.

## 프로젝트 구조
## Project Structure

### 1. Custom_V1.ipynb
- **설명**: AlexNet과 VGG의 특징을 결합한 커스텀 모델의 구현 및 학습.
- **Description**: Implementation and training of a custom model that combines features from AlexNet and VGG.
- **주요 특징**:
  - AlexNet과 VGG의 컨볼루션 레이어의 강점을 결합.
  - 과적합을 줄이기 위해 랜덤 크롭 및 드롭아웃 기법 사용.
  - 레이어별 구현 및 학습에 대한 상세한 코드 포함.
- **Main Features**:
  - Combines the strengths of AlexNet's and VGG's convolutional layers.
  - Uses Random Crop and Dropout techniques to reduce overfitting.
  - Includes detailed code for layer-wise implementation and training.

### 2. Custom_V2.ipynb
- **설명**: ResNet의 잔차 학습 프레임워크를 통합한 커스텀 모델.
- **Description**: A custom model integrating ResNet's residual learning framework into the architecture.
- **주요 특징**:
  - 잔차 연결을 사용해 기울기 소실 문제 완화.
  - 이전 버전 대비 향상된 모델 성능 지표.
  - 커스텀 ResNet 모델 구현을 위한 상세 코드 포함.
- **Main Features**:
  - Residual connections to mitigate the vanishing gradient problem.
  - Improved model performance metrics compared to the previous version.
  - Detailed code for implementing the custom ResNet model.

### 3. Custom_V3.ipynb
- **설명**: Depthwise Separable Convolution을 사용하는 Xception 아키텍처 기반 커스텀 모델.
- **Description**: A custom model based on the Xception architecture with depthwise separable convolutions.
- **주요 특징**:
  - 효율적인 특징 추출을 위한 수정된 Depthwise Separable Convolution 사용.
  - 더 적은 파라미터로 더 나은 성능 달성.
  - Xception 아키텍처의 엔트리, 미들, 엑시트 플로우에 대한 코드 포함.
- **Main Features**:
  - Uses modified Depthwise Separable Convolutions for efficient feature extraction.
  - Achieves better performance with fewer parameters.
  - Contains code for entry, middle, and exit flow of the Xception architecture.

### 4. Custom_V1.pdf
- **설명**: AlexNet과 VGG 아키텍처를 결합한 Custom V1 모델에 대한 설명 자료.
- **Description**: A presentation that explains the Custom V1 model, combining AlexNet and VGG architectures.
- **내용**:
  - AlexNet과 VGG의 아키텍처에 대한 상세 설명.
  - 드롭아웃 및 랜덤 크롭과 같은 과적합 방지 기법.
  - 정확도와 손실 지표를 포함한 학습 결과 시각화.
- **Contents**:
  - Detailed architecture of AlexNet and VGG.
  - Overfitting reduction techniques such as Dropout and Random Crop.
  - Visualization of training results including accuracy and loss metrics.

### 5. Custom_V2.pdf
- **설명**: ResNet의 잔차 블록을 통합한 Custom V2 모델에 대한 설명 자료.
- **Description**: A presentation outlining the Custom V2 model, which integrates ResNet's residual blocks.
- **내용**:
  - ResNet 아키텍처 및 잔차 학습 프레임워크 소개.
  - 커스텀 ResNet 모델 구현 코드 스니펫.
  - 이전 모델들과의 성능 비교.
- **Contents**:
  - Introduction to ResNet architecture and residual learning framework.
  - Code snippets showcasing the implementation of the custom ResNet model.
  - Performance comparison with previous models.

### 6. Custom_V3.pdf
- **설명**: Xception 아키텍처를 활용한 Custom V3 모델에 대한 설명 자료.
- **Description**: A presentation on the Custom V3 model, which leverages the Xception architecture.
- **내용**:
  - Depthwise Separable Convolution 및 그 장점 설명.
  - Xception 아키텍처 개요 및 상세 구현.
  - 이전 버전들과의 모델 성능 비교.
- **Contents**:
  - Explanation of Depthwise Separable Convolution and its advantages.
  - Xception architecture overview and detailed implementation.
  - Comparison of model performance with earlier versions.

### 7. Custom_Conclusion.pdf
- **설명**: 커스텀 모델 실험의 최종 요약.
- **Description**: A final summary of the custom model experiments.
- **내용**:
  - AlexNet, VGG, ResNet, Xception 기반 커스텀 모델들 간의 비교.
  - 각 모델 버전의 성능 지표.
  - 최종 결론 및 향후 작업에 대한 권장 사항.
- **Contents**:
  - Comparison of AlexNet, VGG, ResNet, and Xception based custom models.
  - Performance metrics for each model version.
  - Final thoughts and recommendations for future work.

## 사용 방법
## Usage

1. **환경 설정**:
   - 필요한 Python 패키지 설치: `tensorflow`, `keras`, `numpy`, `matplotlib` 등.
   - 제공된 `requirements.txt` 파일을 사용하여 환경을 설정할 수 있습니다.
   - Install necessary Python packages: `tensorflow`, `keras`, `numpy`, `matplotlib`, etc.
   - You can set up the environment using the provided `requirements.txt` file.

2. **데이터 준비**:
   - ImageNet 데이터셋을 적절히 다운로드하고 지정된 디렉토리에 저장하세요.
   - Jupyter 노트북에서 데이터 경로를 로컬 환경에 맞게 수정하세요.
   - Ensure the ImageNet dataset is properly downloaded and stored in the specified directories.
   - Modify the data paths in the Jupyter notebooks to match your local environment.

3. **모델 학습**:
   - Jupyter 노트북을 실행하여 모델을 학습시키세요.
   - 학습 과정에는 조기 종료 및 모델 체크포인트를 위한 콜백이 포함됩니다.
   - Run the Jupyter notebooks to train the models.
   - The training process includes callbacks for early stopping and model checkpointing.

4. **평가**:
   - 학습이 끝난 후 검증 데이터셋을 사용하여 모델을 평가하세요.
   - 손실과 정확도 그래프를 포함한 학습 히스토리를 시각화하세요.
   - After training, evaluate the models using validation datasets.
   - Visualize the training history including loss and accuracy graphs.

## 결과
## Results

- 각 모델의 성능을 파라미터 수, 손실, 검증 손실, 정확도, 검증 정확도 측면에서 비교했습니다.
- Custom V3 (Xception 기반)은 이전 버전보다 적은 파라미터로 더 나은 성능을 달성했습니다.
- Each model's performance is compared in terms of parameters, loss, validation loss, accuracy, and validation accuracy.
- Custom V3 (Xception-based) achieved better performance with fewer parameters compared to earlier versions.

## 기여
## Contribution

이 프로젝트에 기여하고 싶다면, 리포지토리를 포크한 후 풀 리퀘스트를 제출해 주세요. 문제나 제안 사항이 있다면, 이슈를 열어 주세요.
If you wish to contribute, please fork the repository and submit a pull request. For any issues or suggestions, feel free to open an issue.
