from transformers import pipeline

try: 
   dummy_pipeline = pipeline(task="dummy")

except Exception as e:
   print(e)
