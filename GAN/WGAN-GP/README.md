# [WGAN-GP](https://arxiv.org/pdf/1704.00028) 기반 항공 시뮬레이터의 시각적 품질 향상을 위한 원격 감지 이미지 색상 보정 연구  
# Study on Color Correction of Remote Sensing Images for Visual Quality Enhancement of an Aviation Simulator Based on [WGAN-GP](https://arxiv.org/pdf/1704.00028)

이 프로젝트는 위성 이미지 데이터를 처리하고 분석하는 과정을 포함한 파이프라인을 제공합니다.  
This project provides a pipeline that includes processing and analyzing satellite image data.

프로젝트는 여러 단계로 구성되어 있으며, 각 단계는 Jupyter Notebook 파일로 나뉘어져 있습니다.  
The project consists of several stages, each of which is divided into separate Jupyter Notebook files.

## 파일 목록  
## File List

- `01_Raw to RGB.ipynb`: 원본 위성 이미지 데이터를 RGB 이미지로 변환하는 작업을 수행합니다.  
- `01_Raw to RGB.ipynb`: Converts raw satellite image data into RGB images.

  이 단계에서는 특정 밴드 데이터를 선택하여 RGB 채널로 매핑하여 사용자가 직관적으로 이해할 수 있는 이미지로 변환합니다.  
  In this step, specific band data is selected and mapped to the RGB channels, converting it into an image format that is intuitively understandable.

- `02_Preprocessing.ipynb`: 변환된 RGB 이미지를 전처리하는 단계입니다.  
- `02_Preprocessing.ipynb`: This stage involves preprocessing the converted RGB images.

  이 단계에서는 데이터 정규화, 크기 조정, 이미지 필터링 등의 작업을 수행하여 모델 학습에 적합한 형식으로 데이터를 준비합니다.  
  Tasks such as data normalization, resizing, and image filtering are performed to prepare the data in a format suitable for model training.

- `03_Model_Train.ipynb`: 전처리된 데이터를 사용하여 딥러닝 모델을 학습시키는 단계입니다.  
- `03_Model_Train.ipynb`: This notebook is used for training the deep learning model using the preprocessed data.

  이 노트북에서는 TensorFlow/Keras를 사용하여 모델을 정의하고, 학습을 위한 하이퍼파라미터 설정과 학습 과정의 시각화를 포함하고 있습니다.  
  It defines the model using TensorFlow/Keras, sets hyperparameters for training, and includes visualization of the training process.

- `04_Test.ipynb`: 학습된 모델을 사용하여 테스트 데이터를 예측하는 단계입니다.  
- `04_Test.ipynb`: This stage involves predicting the test data using the trained model.

  이 단계에서는 모델의 성능을 평가하고, 예측 결과를 시각화하여 모델의 정확성을 확인합니다.  
  The performance of the model is evaluated, and the prediction results are visualized to verify the accuracy of the model.

## 사용 방법  
## Usage

1. **환경 설정**: 프로젝트를 실행하기 위해 필요한 라이브러리와 패키지를 설치합니다.  
1. **Setup Environment**: Install the required libraries and packages to run the project.

   주로 사용되는 라이브러리는 `tensorflow`, `numpy`, `matplotlib`, `skimage` 등이 있습니다.  
   Commonly used libraries include `tensorflow`, `numpy`, `matplotlib`, `skimage`, etc.

   필요한 패키지는 `requirements.txt` 파일에 명시되어 있을 수 있으니, 해당 파일을 참고하여 설치하십시오.  
   Refer to the `requirements.txt` file for the necessary packages.

2. **데이터 준비**: 원본 위성 이미지 데이터를 준비합니다.  
2. **Prepare Data**: Prepare the raw satellite image data.

   데이터는 `01_Raw to RGB.ipynb`에서 처리할 수 있는 형식이어야 합니다.  
   The data should be in a format that can be processed by `01_Raw to RGB.ipynb`.

3. **RGB 변환**: `01_Raw to RGB.ipynb`를 실행하여 원본 이미지를 RGB 형식으로 변환합니다.  
3. **RGB Conversion**: Run `01_Raw to RGB.ipynb` to convert the raw images to RGB format.

4. **전처리**: `02_Preprocessing.ipynb`를 실행하여 변환된 이미지를 전처리합니다.  
4. **Preprocessing**: Run `02_Preprocessing.ipynb` to preprocess the converted images.

   이 과정에서 데이터는 학습에 적합한 형식으로 준비됩니다.  
   During this process, the data is prepared in a format suitable for training.

5. **모델 학습**: `03_Model_Train.ipynb`를 실행하여 모델을 학습시킵니다.  
5. **Model Training**: Run `03_Model_Train.ipynb` to train the model.

   이 단계에서는 전처리된 데이터를 사용하여 딥러닝 모델을 학습하게 됩니다.  
   This step involves training the deep learning model using the preprocessed data.

6. **모델 테스트**: `04_Test.ipynb`를 실행하여 학습된 모델을 평가하고, 테스트 데이터를 통해 모델의 성능을 확인합니다.  
6. **Model Testing**: Run `04_Test.ipynb` to evaluate the trained model and verify the performance of the model using test data.

## 주의 사항  
## Notes

- 각 노트북은 순차적으로 실행되어야 하며, 이전 단계의 출력 데이터를 다음 단계의 입력 데이터로 사용합니다.  
- Each notebook should be run sequentially, as the output data from the previous step is used as input for the next step.

- GPU를 사용하는 경우, 노트북에서 GPU 설정을 확인하고 필요한 경우 환경을 설정하십시오.  
- If using a GPU, check the GPU settings in the notebooks and configure the environment if necessary.

## 기여  
## Contribution

이 프로젝트에 기여하고 싶다면, Pull Request를 제출해 주세요.  
If you would like to contribute to this project, please submit a Pull Request.

버그 리포트나 개선 사항 제안도 환영합니다.  
Bug reports and suggestions for improvements are also welcome.