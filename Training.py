import pandas as pd
import os
import time
import datetime

from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel
import torch
from torch.utils.data import Dataset, DataLoader
# import pytorch_lighting as pl
from sklearn.model_selection import train_test_split
import numpy as np
import random
import textwrap
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead

# tokenizer = AutoTokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>',
#                                           pad_token='<|pad|>')
# torch.cuda.set_device(0)
# torch.cuda.current_device()
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.save_pretrained('Recipe')
tokenizer.add_special_tokens({
    'eos_token': '<EOR>',
    'additional_special_tokens': [
        '<SOI>',
        '<ISEP>',
        '<EOI>',
        '<SOR>']
})
tokenizer.truncation = True
tokenizer.max_length = 512
model = AutoModelForCausalLM.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, truncation=True, return_tensors='pt', max_length=512)
output = model(**encoded_input)

train_path = 'train_dataset2.txt'
test_path = 'test_dataset2.txt'


def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=16)

    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=16)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
    )
    return train_dataset, test_dataset, data_collator


train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)
training_args = TrainingArguments(
    output_dir="./gpt2",  # The output directory
    overwrite_output_dir=True,  # overwrite the content of the output directory
    num_train_epochs=1,  # number of training epochs
    per_device_train_batch_size=32,  # batch size for training
    per_device_eval_batch_size=32,  # batch size for evaluation
    eval_steps=6400,  # Number of update steps between two evaluations.
    save_steps=6400,  # after # steps model is saved
    warmup_steps=1000,  # number of warmup steps for learning rate scheduler
    prediction_loss_only=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.is_model_parallel = True

trainer.train()
trainer.save_model()
