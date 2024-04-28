import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# 图片艺术评分模型
# https://github.com/christophschuhmann/improved-aesthetic-predictor
# 不要改动模型结构代码，除非自己重新训练
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
# 推理机器环境
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
clip_model_name = "openai/clip-vit-large-patch14"
clipModel = CLIPModel.from_pretrained(clip_model_name).to(device)
clipProcessor = CLIPProcessor.from_pretrained(clip_model_name)

# 加载评分模型
rating_model_name = "sac+logos+ava1-l14-linearMSE.pth"
mlpModel = MLP(768)
s = torch.load(rating_model_name, map_location=device)
mlpModel.load_state_dict(s)
mlpModel.to(device)
mlpModel.eval()
    
def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def getImageByUrl(img_url: str):
    return Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

def getImageFeaturesByCLIP(image):
    inputs = clipProcessor(images=image, return_tensors="pt").to(device)
    return clipModel.get_image_features(**inputs)

def getAestheticScore(img_url: str):
    image = getImageByUrl(img_url)
    image_features = getImageFeaturesByCLIP(image)
    with torch.no_grad():
        if device == 'cuda':
            im_emb_arr = normalized(image_features.detach().cpu().numpy())
            im_emb = torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor)
        else:
            im_emb_arr = normalized(image_features.detach().numpy())
            im_emb = torch.from_numpy(im_emb_arr).to(device).type(torch.FloatTensor)

        prediction = mlpModel(im_emb)

    score = prediction.item()
    print(score)

test_url = 'https://cdn.bebememo.us/alijp/pictures/original/202212/537617569/4f5e40f19663442ab6e2eaffb2fcbf4a.jpg!large'
getAestheticScore(test_url)