import torch

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
