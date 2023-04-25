# clip_proj

- The goal of the project is to implement an image retrieval system that uses the CLIP model's similarity metric to find the most similar image and text for a given image. The model is trained on the Flickr 30k dataset, which consists of 31,783 images and captions, with each image having five captions that describe the image. The dataset is split into training and testing sets in an 80:20 ratio. The implementation is designed to handle multiple image inputs simultaneously.

- The system is built using PyTorch and Hugging Face library, which is a transformer-based framework that leverages pre-trained deep learning models. The system is composed of two major components: training and evaluation on the dataset of the implementation of the CLIP model using BERT and ViT models for text and image embeddings, respectively, and finally, a RESTful inference API.

- The resulting implementation comprises an inference API and an inference script that uses curl to trigger the inference API. The script accepts one or more image files as input, sends them to the API, and as an output the most similar image and text for each image is returned. In addition, the script implements caching for the inference results to reduce the API's response time. 

## Setup 

### Download Dataset :

Downlaod dataset named flickr30k_images from the kaggle.

### Requirements 

- [x] albumentations
- [x] Flask==2.1.0
- [x] flask_cors==3.0.10
- [x] Flask-Caching
- [x] matplotlib
- [x] numpy
- [x] opencv-python
- [x] pandas
- [x] pickle5
- [x] pillow
- [x] timm
- [x] torch==1.11.0 -f https://download.pytorch.org/whl/torch_stable.html
- [x] transformers==4.25.1
- [x] tqdm

- after cloning  you just have to download two models named , clip_bert_vip.pt , pytorch_model.bin from gDrive and add them to the folder named “models” .
Gdrive link : https://drive.google.com/drive/folders/1u2LmgzCWTIcdMW6q_XiE4cn-w7_wFsQB?usp=share_link


### commands 
```sh
$ git clone https://github.com/shubhamMehla12/clip_proj.git  
$ cd clip_proj

$ pip install -r requirements.txt

$ cd src
$ python main.py
```
and your Inference API is ready!

## Input : 

![image](https://user-images.githubusercontent.com/109681358/234265278-14af0a7b-d0e6-4dee-b2ef-0297beb93aed.png)

## output : 

![image](https://user-images.githubusercontent.com/109681358/234265337-6a07cee0-2266-4082-848a-a2ccd4c25a2a.png)

![image](https://user-images.githubusercontent.com/109681358/234265364-ca0a2f23-8572-4b10-8698-71aab2194115.png)


