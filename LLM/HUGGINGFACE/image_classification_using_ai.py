# Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("object-detection", model="facebook/detr-resnet-50")


# Load model directly
from transformers import DetrImageProcessor, DetrForObjectDetection
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

model.eval()
print(model.device)

model.config.id2label


# Preparing the Image
from PIL import Image, ImageDraw
import requests
import torch
import subprocess
import os

def load_image(url):
   if url.startswith('http'):
       image = Image.open(requests.get(url, stream=True).raw)
   else:
       image = Image.open(url)
   return image

url_image = 'https://scholar.uc.edu/downloads/2801ph76n?locale=en'
image = load_image('https://scholar.uc.edu/downloads/2801ph76n?locale=en')
# subprocess.run(['open', url_image])


# Detecting Objects

inputs =image_processor(images = image, return_tensors = "pt")
outputs = model(**inputs)
print(outputs)

target_sizes = torch.tensor([image.size[::-1]])

results = image_processor.post_process_object_detection(outputs,target_sizes = target_sizes, threshold = 0.9)[0]
#print(results)

# Draw Image

draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
   box= [round(i,2) for i in box.tolist()]
   print(
       f"Detected{model.config.id2label[label.item()]} with confidence"
       f"{round(score.item(),3)} at location {box}"
   )

   draw.rectangle(box,outline="yellow", width=15)

   draw.text((box[0], box[1]-20),model.configid2label[label.item()],fill="white")

image

updated_image_path = 'updated_image.jpg'
image.save(updated_image_path)

current_directory = os.getcwd()

# Construct the full path
file_path = os.path.join(current_directory, updated_image_path)

print(file_path)
# subprocess.run(['open', file_path])
