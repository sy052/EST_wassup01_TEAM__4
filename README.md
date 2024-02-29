# SMART PIXCELS
![image](https://github.com/Bong-HoonLee/EST_wassup01_TEAM__4/assets/115579916/8d2f8a80-8fc2-4012-b17d-7a498b4c23c4)


## Non-face-to-face psychological counseling service using AI human
- Purpose of the project
	- Establishing a business plan using an Facial Expression Recognition classification model
  - Non-face-to-face psychological counseling treatment
  - Relieves social isolation by feeling like talking to real people
 
- A major customer base
  1) A person with social phobia
  2) A person with depression
  3) Elder who lives alone
  4) someone in need of a conversation

### Team
- [Bong-HoonLee](https://github.com/Bong-HoonLee)
- [Jae-SeokLee](https://github.com/appleman153)
- [Sejun Choi](https://github.com/enversel)
- [SoyeonHwang](https://github.com/sy052)

## 개발 환경
| **IDE**           | **GPU 서버**                    | **프로그래밍 언어** |
| ----------------- | ------------------------------- | ------------------- |
| ![VSCode](https://img.shields.io/badge/Cursor-Visual%20Studio%20Code-blue?style=for-the-badge&logo=visual-studio-code) | ![A100 GPU](https://img.shields.io/badge/Microsoft%20Azure-A100%20GPU-0078D4?style=for-the-badge&logo=microsoft-azure)       | ![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)              |


### Directory
- `.streamlit` : Folder used for streamlit
- `archive`: EDA, data pre-processing
- `config_f`: Auto-training config folder
- `contents` : source
- `docs`: documents, images, reports
- `models`: models
- `tools` : Other Architectures
- `utils` : Metric and other files
- `requirements.txt`: required libraries and packages 
- `trainer.py`: main train&test logics

## How to Run & Debug
1) `pip install -r requirements.txt` to install required packages
2) streamlit run main.py
3) Drag image to sidebar uploading
4) And Try Psychological counseling


## Dataset
- Psychological counseling paper in severance hospital

## EDA
- 

## Models
- You can check the list at config.py
1) List of Neural Network models used to train models (total:18)
- alexnet		
- convnext_tiny	
- densenet121	
- efficientnet_v2_s	
- googlenet	
- inception_v3		
- mnasnet0_5		
- mobilenet_v3_large	
- resnet18		
- resnet34		
- resnet50		
- resnet101	
- vgg11_bn		
- vgg13_bn	
- vgg16_bn	
- vit_b_16	
- swin_t	
- custom

2) Finally selected neural network models (total:10)
: Select by considering the appropriate model and performance for the chatbot
- Yolo v8
- AlexNet
- DenseNet121
- EfficientNet
- VGG
- ResNet
- ViT
- swin_t
- MobileNet
- Custom model

# result
- we chose the YOLO with the fastest and the highest accuracy<br><br>
![image](https://github.com/Bong-HoonLee/EST_wassup01_TEAM__4/assets/54875204/b65091f7-1615-420d-b854-66089aed025e)
![image](https://github.com/Bong-HoonLee/EST_wassup01_TEAM__4/assets/54875204/12033bf3-1274-4e15-895c-319df51621ea)
- prams : 3.2M
- Accuracy: 69%

![image](https://github.com/Bong-HoonLee/EST_wassup01_TEAM__4/assets/115579916/568470df-1572-47a3-a5c1-3a2b8ffc77cc)

# service

![image](https://github.com/Bong-HoonLee/EST_wassup01_TEAM__4/assets/115579916/1c7b9e27-bb62-4aad-b433-4d2825dd3bd5)



