import requests
import torch
import time

from PIL import Image
from fastapi import FastAPI
from transformers import CLIPProcessor, CLIPModel
    
# 推理机器环境
device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载 CLIP 模型
clip_model_name = "openai/clip-vit-large-patch14"
clipModel = CLIPModel.from_pretrained(clip_model_name).to(device)
clipProcessor = CLIPProcessor.from_pretrained(clip_model_name)

tags = [
    "a photo with no child at all",
    "a photo with only one child",
    "a photo with two children",
    "a photo with three children",
    "a photo with many children",
    "a photo with frontal face",
    "a photo with side face",
    "a photo with obscured face",
    "a photo with smiling face",
    "a photo with crying face",
    "a photo with calming face"
]

def getImageByUrl(img_url: str):
    return Image.open(requests.get(img_url, stream=True).raw).convert('L')

def getMaxProbTag(image, input_tags):
    threshold = 0

    inputs = clipProcessor(text=input_tags, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clipModel(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).tolist()[0]

    results = []
    for index,tag in enumerate(input_tags):
        if probs[index] >= threshold:
            results.append({
                "name": tag.replace('a photo with ',''),
                "confidence": probs[index]
            })
    
    return results

def getAnalysis(
    image, 
    show_tags: bool = False
):
    outputs = getMaxProbTag(image, tags)

    sum_child = 0
    sum_face = 0
    sum_expression = 0
    for i, item in enumerate(outputs):
        if i < 5:
            sum_child += item["confidence"]
        elif i < 8:
            sum_face += item["confidence"]
        else:
            sum_expression += item["confidence"]

    for i, item in enumerate(outputs):
        if i < 5:
            item["confidence"] = round(item["confidence"] / sum_child, 3)
        elif i < 8:
            item["confidence"] = round(item["confidence"] / sum_face, 3)
        else:
            item["confidence"] = round(item["confidence"] / sum_expression, 3)
    
    score_final = 0
    for i, item in enumerate(outputs):
        if item["name"] == "only one child":
            score_child = min(item["confidence"], 0.6)
            score_final += score_child
        if item["name"] == "frontal face":
            score_face = min(item["confidence"], 0.6)
            if score_child == 0.6:
                score_final += score_face
        if item["name"] == "smiling face":
            score_expression = min(item["confidence"], 0.8)
            if score_child == 0.6 and score_face == 0.6 :
                score_final += score_expression

    result = {}
    result["score"] = score_final
    if show_tags:
        result["tags"] = outputs

    return result

app = FastAPI()

@app.get("/")
async def root():
    return "BebeRatingPistol Api"

@app.get("/rate")
async def getRating(
    img_url: str,
    show_tags: bool = False
):
    t1 = time.time()
    result = {}

    try:
        image = getImageByUrl(img_url)
    except IOError as e:
        print("下载异常:", e)
        return {
            "score" : 0,
            "error" : "image download fail"
        }

    t2 = time.time()
    print(f"下载时间: {round(t2 - t1, 3)} 秒")

    try:
        result = getAnalysis(image, show_tags)
    except Exception as e:
        print("推理异常:", e)
        return {
            "score" : 0,
            "error" : "model inference fail"
        }

    t3 = time.time()
    print(f"推理时间: {round(t3 - t2, 3)} 秒")

    return result