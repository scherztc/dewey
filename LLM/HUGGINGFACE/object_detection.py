import gradio as gr
from PIL import Image, ImageDraw
from transformers import pipeline

# Initialize the object detection pipeline
objDetector = pipeline(task="object-detection", model="facebook/detr-resnet-50")

def detect_objects(image):
    # Convert the input image from Gradio to a PIL Image
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    # Perform object detection
    results = objDetector(image)

    for obj in results:
        if obj['score'] < 0.9:
            continue

        # Extract bounding box coordinates
        box = [
            obj['box']['xmin'],
            obj['box']['ymin'],
            obj['box']['xmax'],
            obj['box']['ymax']
        ]

        # Draw rectangle and label on the image
        draw.rectangle(box, outline="yellow", width=5)
        draw.text((box[0], box[1] - 15), obj['label'], fill='white')

    return image

# Create the Gradio interface
demo = gr.Interface(
    fn=detect_objects,
    inputs=gr.Image(label="Upload Image"),
    outputs=gr.Image(label="Detected Objects")
)

# Launch the Gradio app
demo.launch(share=True)
