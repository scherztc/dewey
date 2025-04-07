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

