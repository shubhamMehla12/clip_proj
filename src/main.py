import os
import cv2
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizer
import pickle
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from flask import Flask, request, jsonify
from flask_caching import Cache 
import config as CFG
from models import CLIPModel

cache = Cache()

app = Flask(__name__)
app.config['CACHE_TYPE'] = 'simple'

app.config['CACHE_TYPE'] = 'simple'
cache.init_app(app)

model_vit_path = "../models/vit" 
model_vit = VisionEncoderDecoderModel.from_pretrained(model_vit_path)
feature_extractor_vit = ViTImageProcessor.from_pretrained(model_vit_path)
tokenizer_vit = AutoTokenizer.from_pretrained(model_vit_path)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_vit.to(device)

def predict_step(image_paths):
  max_length = 16
  num_beams = 4
  gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor_vit(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model_vit.generate(pixel_values, **gen_kwargs)

  preds = tokenizer_vit.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

def make_train_valid_dfs():
    dataframe = pd.read_csv("../flickr30k_images/captions.csv")
    
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    return train_dataframe, valid_dataframe

def get_captions(query,names):
    captions = {}
    for name in names:
        df = pd.read_csv("../flickr30k_images/captions.csv")
        df = df[df['image']==name].reset_index(drop=True)
        captions_lst = [df['caption'].iloc[idx] for idx in range(len(df))]
        # Initialize CountVectorizer
        vectorizer = CountVectorizer().fit_transform([query] + captions_lst)

        # Calculate cosine similarity
        cosine_similarities = cosine_similarity(vectorizer[0], vectorizer[1:]).flatten()

        # Get the most similar text
        most_similar_text = captions_lst[cosine_similarities.argmax()]
        
        captions[name] = most_similar_text
    return captions


def find_matches(model, image_embeddings, query, image_filenames,file_name, n=9):
    make_dir(f"outputs/{file_name}")

    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    encoded_query = tokenizer([query])
    batch = {
        key: torch.tensor(values).to(CFG.device)
        for key, values in encoded_query.items()
    }
    with torch.no_grad():
        text_features = model.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        text_embeddings = model.text_projection(text_features)
    
    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embeddings_n = F.normalize(text_embeddings, p=2, dim=-1)
    dot_similarity = text_embeddings_n @ image_embeddings_n.T
    
    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]
    
    captions = get_captions(query,matches)
    for match in matches:
        image = cv2.imread(f"{CFG.image_path}/{match}")
        cv2.imwrite(f"../outputs/{file_name}/{match}",image)
    
    return captions

def load_model_n_image_emb(model_path):
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()

    # Open the file for reading in binary mode
    with open('../models/image_embeddings_cpu.pkl', 'rb') as f:
        # Use the pickle.load() method to read the matrix from the file
        image_embeddings = pickle.load(f)
        
    return model,image_embeddings

def make_dir(name):
    try:
        os.mkdir(f"../{name}")
    except:
        pass

def delete_folder(path):
    try:
        for root, dirs, files in os.walk(f"../{path}", topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(f"../{path}")
    except:
        pass

@app.route('/predict', methods=['POST'])
@cache.cached(timeout=60)
def predict():
    files = request.files.getlist('file')

    make_dir("inputs")
    delete_folder("outputs")
    filepaths = []
    for file in files:
        name = file.filename
        filepaths.append(f"../inputs/{name}.jpg")
        file.save(f"../inputs/{name}.jpg")
    
    _, valid_df = make_train_valid_dfs()
    model,image_embeddings = load_model_n_image_emb(CFG.model_path)

    queries = predict_step(filepaths)
    make_dir("outputs")
    output = {}
    for idx,file in enumerate(files):
        name = file.filename
        matches = find_matches(model, 
                image_embeddings,
                query=queries[idx],
                image_filenames=valid_df['image'].values,
                file_name = name,
                n=5)
        output[name[:-4]] = {"caption":queries[idx],"matches":matches}
    return jsonify({"image_path":CFG.image_path,"output":output})

    
if __name__ == '__main__':
    app.run(debug=True)
