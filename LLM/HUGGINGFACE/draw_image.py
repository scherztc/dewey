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
