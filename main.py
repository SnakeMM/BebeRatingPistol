import requests
import torch
import time

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
    #results = sorted(results, key=lambda k: k["confidence"], reverse=True)
    #print(results)
    return results

def getAnalysis(img_url: str):
    t1 = time.time()

    image = getImageByUrl(img_url)

    t2 = time.time()
    print(f"下载时间: {round(t2 - t1, 3)} 秒")

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

    outputs = getMaxProbTag(image, tags)

    t3 = time.time()
    print(f"推理时间: {round(t3 - t2, 3)} 秒")

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
    result["tags"] = outputs
    result["score_final"] = round(score_final * 50, 1)
    print(result)

    return result

getAnalysis('https://cdn.bebememo.us/alijp/pictures/original/202405/537057801/37dc60f3f81045639810017dd43c6b1f.jpg!large')

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
        i = 0
        for item in list:
            result = {}
            result["url"] = item
            result["detail"] = getAnalysis(item)
            results.append(result)
            i += 1
            print(f"处理图片: {i} / {len(list)}")

        results = sorted(results, key=lambda k: k["detail"]["score_final"], reverse=True)
        return results