import gradio as gr
from PIL import Image, ImageDraw
from transformers import pipeline

# Initialize the object detection pipeline



pipe = pipeline(task="image-segmentation", model="nvidia/segformer-b0-finetuned-ade-512-512")

def detect_objects(image):
    # Convert the input image from Gradio to a PIL Image
    image = Image.fromarray(image)
#    draw = ImageDraw.Draw(image)

    # Perform object detection
    results = pipe(image)

    for result in results:
        if result['label'] == 'person':
            print('found person')
            base_image = image.copy()
            mask_image = result ['mask']
	    mask_iamge = ImageOps.invert(mask_image)
            base_image.paste(mask_image,mask=mask_image)
            return base_image

