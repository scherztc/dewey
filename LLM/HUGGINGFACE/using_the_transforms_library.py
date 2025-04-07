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

def load_image(url):
   if url.startswith('http'):
       image = Image.open(requests.get(url, stream=True).raw)
   else:
       image = Image.open(url)
   return image

url_image = 'https://scholar.uc.edu/downloads/2801ph76n?locale=en'
image = load_image('https://scholar.uc.edu/downloads/2801ph76n?locale=en')
# subprocess.run(['open', url_image])
