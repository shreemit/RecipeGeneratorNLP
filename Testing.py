

from transformers import pipeline

chef = pipeline('text-generation',model='./gpt2', tokenizer='gpt2', max_length = 400)

print(chef('Chicken, salt, pepper')[0]['generated_text'])

