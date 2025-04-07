from transformers import pipeline

translator = pipeline('translation_en_to_fr')

print(translator('Soccer is the greatest sport in the World.'))


