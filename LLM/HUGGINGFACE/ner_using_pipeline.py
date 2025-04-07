from transformers import pipeline

import pandas as pd

text = '''
Thomas Scherz is a lead software developer at the University of Cincinnati.
'''

ner = pipeline('ner',aggregation_strategy = 'simple')
df = pd.DataFrame(ner(text))
print(df)
