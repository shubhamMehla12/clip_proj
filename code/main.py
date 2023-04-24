import os
import cv2
import gc
import numpy as np
import pandas as pd
import itertools
from tqdm.autonotebook import tqdm
import albumentations as A
import matplotlib.pyplot as plt

import torch
from torch import nn
import torch.nn.functional as F
import timm
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import pickle
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)


model_vit_path = "../models/vit" 
model_vit = VisionEncoderDecoderModel.from_pretrained(model_vit_path)
feature_extractor_vit = ViTImageProcessor.from_pretrained(model_vit_path)
tokenizer_vit = AutoTokenizer.from_pretrained(model_vit_path)

# model_vit.save_pretrained("../models/vit/main")
# feature_extractor_vit.save_pretrained("../models/vit/main")
# tokenizer_vit.save_pretrained("../models/vit/main")

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


class CFG:
    debug = False
    image_path = "../flickr30k_images/flickr30k_images"
    model_path = "../models/clip_bert_vit.pt" #./models/best.pt"
    captions_path = "."
    batch_size = 32
    num_workers = 4
    head_lr = 1e-3
    image_encoder_lr = 1e-4
    text_encoder_lr = 1e-5
    weight_decay = 1e-3
    patience = 1
    factor = 0.8
    epochs = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "vit_base_patch32_plus_256"#"vit_small_resnet50d_s16_224"#"vit_relpos_medium_patch16_224"#'resnet50'
    image_embedding = 896
    text_encoder_model = "distilbert-base-uncased"
    text_embedding = 768
    text_tokenizer = "distilbert-base-uncased"
    max_length = 200

    pretrained = True # for both image encoder and text encoder
    trainable = True # for both image encoder and text encoder
    temperature = 1.0

    # image size
    size = 256
    

    # for projection head; used for both image and text encoders
    num_projection_layers = 1
    projection_dim = 224 
    dropout = 0.1


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]

class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """

        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=CFG.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{CFG.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item


    def __len__(self):
        return len(self.captions)



def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(CFG.size, CFG.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )

class ImageEncoder(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
        self, model_name=CFG.model_name, pretrained=CFG.pretrained, trainable=CFG.trainable
    ):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)

class TextEncoder(nn.Module):
    def __init__(self, model_name=CFG.text_encoder_model, pretrained=CFG.pretrained, trainable=CFG.trainable):
        super().__init__()
        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            self.model = DistilBertModel(config=DistilBertConfig())
            
        for p in self.model.parameters():
            p.requires_grad = trainable

        # we are using the CLS token hidden representation as the sentence's embedding
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = output.last_hidden_state
        return last_hidden_state[:, self.target_token_idx, :]

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=CFG.projection_dim,
        dropout=CFG.dropout
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class CLIPModel(nn.Module):
    def __init__(
        self,
        temperature=CFG.temperature,
        image_embedding=CFG.image_embedding,
        text_embedding=CFG.text_embedding,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(embedding_dim=image_embedding)
        self.text_projection = ProjectionHead(embedding_dim=text_embedding)
        self.temperature = temperature

    def forward(self, batch):
        # Getting Image and Text Features
        image_features = self.image_encoder(batch["image"])
        text_features = self.text_encoder(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings = self.image_projection(image_features)
        text_embeddings = self.text_projection(text_features)

        # Calculating the Loss
        logits = (text_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        texts_similarity = text_embeddings @ text_embeddings.T
        targets = F.softmax(
            (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
        )
        texts_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def make_train_valid_dfs():
    # df = pd.read_csv("../flickr30k_images/results.csv",sep="|")
    # df.columns = ['image', 'caption_number', 'caption']
    # df['caption'] = df['caption'].str.lstrip()
    # df['caption_number'] = df['caption_number'].str.lstrip()
    # df.loc[19999, 'caption_number'] = "4"
    # df.loc[19999, 'caption'] = "A dog runs across the grass ."
    # ids = [id_ for id_ in range(len(df) // 5) for i in range(5)]
    # df['id'] = ids
    # df.to_csv("../flickr30k_images/captions.csv", index=False)

    dataframe = pd.read_csv("../flickr30k_images/captions.csv")
    
    # dataframe = dataframe.iloc[:10,:]
    max_id = dataframe["id"].max() + 1 if not CFG.debug else 100
    image_ids = np.arange(0, max_id)
    np.random.seed(42)
    valid_ids = np.random.choice(
        image_ids, size=int(0.2 * len(image_ids)), replace=False
    )
    train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
    train_dataframe = dataframe[dataframe["id"].isin(train_ids)].reset_index(drop=True)
    valid_dataframe = dataframe[dataframe["id"].isin(valid_ids)].reset_index(drop=True)
    
    # train_dataframe = dataframe.iloc[:5,:]
    # valid_dataframe = dataframe.iloc[:3,:]
    # print(train_dataframe.shape,valid_dataframe.shape)
    
    return train_dataframe, valid_dataframe


def build_loaders(dataframe, tokenizer, mode):
    transforms = get_transforms(mode=mode)
    dataset = CLIPDataset(
        dataframe["image"].values,
        dataframe["caption"].values,
        tokenizer=tokenizer,
        transforms=transforms,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True if mode == "train" else False,
    )
    return dataloader

def get_image_embeddings(valid_df, model_path):
    tokenizer = DistilBertTokenizer.from_pretrained(CFG.text_tokenizer)
    valid_loader = build_loaders(valid_df, tokenizer, mode="valid")
    
    model = CLIPModel().to(CFG.device)
    model.load_state_dict(torch.load(model_path, map_location=CFG.device))
    model.eval()
    
    valid_image_embeddings = []
    with torch.no_grad():
        for batch in tqdm(valid_loader):
            image_features = model.image_encoder(batch["image"].to(CFG.device))
            image_embeddings = model.image_projection(image_features)
            valid_image_embeddings.append(image_embeddings)
    return model, torch.cat(valid_image_embeddings)

def get_captions(names):
    captions = {}
    for name in names:
        df = pd.read_csv("../flickr30k_images/captions.csv")
        df = df[df['image']==name].reset_index(drop=True)
        caption = df['caption'].iloc[0]
        captions[name] = caption
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
    
    captions = get_captions(matches)
    # _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match in matches:
        image = cv2.imread(f"{CFG.image_path}/{match}")
        cv2.imwrite(f"../outputs/{file_name}/{match}",image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     ax.imshow(image)
    #     ax.axis("off")
    # plt.show()
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
    # print(queries)
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
        output[name[:-4]] = {"query":queries[idx],"matches":matches}
    return jsonify({"output":output})

    
if __name__ == '__main__':
    app.run(debug=True)
