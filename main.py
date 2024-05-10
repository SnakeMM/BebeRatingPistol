import requests
import torch

from PIL import Image
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel
    
# 推理机器环境
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
clip_model_name = "openai/clip-vit-large-patch14"
clipModel = CLIPModel.from_pretrained(clip_model_name).to(device)
clipProcessor = CLIPProcessor.from_pretrained(clip_model_name)

def getImageByUrl(img_url: str):
    return Image.open(requests.get(img_url, stream=True).raw).convert('L')

def getMaxProbTag(image, input_tags):
    threshold = 0.01

    inputs = clipProcessor(text=input_tags, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clipModel(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]

    results = []
    for index,tag in enumerate(input_tags):
        if probs[index] >= threshold:
            results.append({
                "name": tag.replace('a photo with ',''),
                "confidence": round(probs[index],2)
            })
    results = sorted(results, key=lambda k: k["confidence"], reverse=True)
    #print(results)
    return results

def getAnalysis(img_url: str):
    result = {}
    image = getImageByUrl(img_url)

    tags_child = [
        "a photo with no child at all",
        "a photo with only one child",
        "a photo with two children",
        "a photo with three children",
        "a photo with many children"
    ]
    tags_face = [
        "a photo with front face",
        "a photo with side face",
        "a photo with no face"
    ]
    tags_expression = [
        "a photo with smiling face",
        "a photo with crying face",
        "a photo with calming face"
    ]

    score_child = 0
    score_face = 0
    score_expression = 0
    score_final = 0

    result["tags_child"] = getMaxProbTag(image, tags_child)
    for item in result["tags_child"]:
        if item["name"] == "only one child":
            score_child = item["confidence"]
            score_final += score_child

    result["tags_face"] = getMaxProbTag(image, tags_face)
    for item in result["tags_face"]:
        if item["name"] == "front face":
            score_face = item["confidence"]
            if score_child > 0.6:
                score_final += score_face

    result["tags_expression"] = getMaxProbTag(image, tags_expression)
    for item in result["tags_expression"]:
        if item["name"] == "smiling face":
            score_expression = item["confidence"]
            if score_child > 0.6 and score_face > 0.6 :
                score_final += score_expression

    result["score_final"] = score_final

    return result

#getAnalysis('https://cdn.bebememo.us/alijp/pictures/original/202405/537617569/09134e0efd0f400da4d43559ee0b9e03.jpg!large')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return "BebeRatingPistol Api"

@app.get("/rate/batch")
async def getTagsClip(
    baby_id: str, 
    total_days: str
):
    url = f"https://beta.bebememo.us/tests?baby_id={baby_id}&total_days={total_days}"
    headers = {
        "Authorization": "us_537062423_AZoBbN9IKFMGMETGK34BBtQv1ioRhYi4XSeErEVim7Q"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        list = data["list"]
        results = []
        for item in list:
            result = {}
            result["url"] = item
            result["detail"] = getAnalysis(item)
            results.append(result)

        results = sorted(results, key=lambda k: k["detail"]["score_final"], reverse=True)
        return results