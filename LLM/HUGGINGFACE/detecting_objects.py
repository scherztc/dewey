# Detecting Objects

inputs =image_processor(images = image, return_tensors = "pt")
outputs = model(**inputs)
print(outputs)

target_sizes = torch.tensor([image.size[::-1]])

results = image_processor.post_process_object_detection(outputs,target_sizes = target_sizes, threshold = 0.9)[0]
#print(results)
