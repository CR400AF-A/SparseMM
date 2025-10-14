import os
import io
import json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image, ImageDraw

# load dataset
dataset_path = "/path/to/datasets/coco/data"
data = load_dataset(dataset_path, split="validation")

# set save_path
image_folder = "/path/to/datasets/coco/image"
json_folder = "/path/to/datasets/coco"

# mkdir
if not os.path.exists(image_folder):
    os.makedirs(image_folder)
if not os.path.exists(json_folder):
    os.makedirs(json_folder)

# data_info
with open("/path/to/datasets/coco/dataset_infos.json", "r") as f:
    data_info = json.load(f)

category_info = data_info['detection-datasets--coco']['features']["objects"]['feature']['category']['names']


converted_data = []

for id, da in enumerate(tqdm(data)):
    json_data = {}
    json_data["id"] = da['image_id']
    if da["image"] is not None:
        json_data["image_name"] = f"/path/to/datasets/coco/image/{da['image_id']}.jpg"
        image = Image.open(io.BytesIO(da["image"]["bytes"]))
        image.save(json_data["image_name"])

    objects = da["objects"]
    category = []
    for i in range(len(objects['category'])):
        category.append(category_info[objects["category"][i]])
    objects['category'] = category
    json_data["content"] = objects
    converted_data.append(json_data)
    
with open("/path/to/datasets/coco/coco_val.json", "w") as f:
    json.dump(converted_data, f, indent=4)

print("data process done!")