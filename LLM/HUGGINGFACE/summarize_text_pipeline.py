from transformers import pipeline

import pandas as pd

text = '''
Thomas Scherz is a lead software developer at the University of Cincinnati.  He also works for Kolping and Turfway race track.
'''

text_summarizer = pipeline('summarization')
output = text_summarizer(text, max_length = 260, clean_up_tokenization_spaces = True)

print(output[0]['summary_text'])
