from transformers import pipeline

classifier = pipeline(task ="text-classification")

# classifier = pipeline(task = "text-classification", model = "distilbert-base-uncased-finetuned-sst-2-english", revision = "af0f9b")

import pandas as pd

text = '''
I thought this was a wonderful summer day.
'''

result = classifier(text)
df = pd.DataFrame(result)
print(df)

