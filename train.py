import os
import torch
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
# from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, GPT2LMHeadModel, AdamW
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.model_selection import train_test_split
from transformers import get_scheduler
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import wandb

class RecipeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.dataframe = dataframe
        self.ingred = self.dataframe.ingred
        self.instructions = self.dataframe.instructions
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '<PAD>',
                                           'additional_special_tokens': ['<I>']})
        self.source_len = source_len
        self.summ_len = summ_len

    def __len__(self):
        return len(self.dataframe)


    def __getitem__(self, idx):
        text = str(self.ingred[idx])
        ctext = str(self.instructions[idx])

        source = self.tokenizer.batch_encode_plus([text], 
                                                  max_length=self.source_len, 
                                                  padding='max_length', 
                                                  return_tensors='pt', 
                                                  truncation=True)
        
        target = self.tokenizer.batch_encode_plus([ctext], 
                                                  max_length=self.summ_len, 
                                                  padding='max_length', 
                                                  return_tensors='pt', 
                                                  truncation=True)

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_mask.to(dtype=torch.long)
        }
import wandb

class T5Trainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        val_dataloader,
        device,
        epochs=3,
        learning_rate=5e-5,
        warmup_steps=500,
        epsilon=1e-8,
        sample_every=100,
        max_grad_norm=1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model.resize_token_embeddings(len(tokenizer))
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.epsilon = epsilon
        self.sample_every = sample_every
        self.optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate)
        num_training_steps = epochs * len(train_dataloader)
        self.lr_scheduler = get_scheduler(name="linear", 
                                          optimizer=self.optimizer, 
                                          num_warmup_steps=self.warmup_steps, 
                                          num_training_steps=num_training_steps)
        self.max_grad_norm = max_grad_norm

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            print("Training", self.train_dataloader.dataset.__len__())
            for iteration, data in tqdm(enumerate(self.train_dataloader)):

                batch = {k: v.to(self.device) for k, v in data.items()}
                outputs = self.model(input_ids = batch['source_ids'], 
                                     attention_mask=batch['source_mask'],
                                     labels=batch["target_ids"])
                loss = outputs.loss
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                if iteration % self.sample_every == 0:
                    print(f'Epoch: {epoch}, Iteration: {iteration}, Loss: {loss.item()}')
                    # Log metrics to Weights & Biases
                    wandb.log({"loss": loss.item()})

                    # # Calculate and log BLEU and ROUGE scores
                    # self.model.eval()
                    # self.model = self.model.to('cpu')
                    # batch = {k: v.to('cpu') for k, v in data.items()}
                    # predictions = self.model.generate(batch['source_ids'], max_length=128, num_beams=4, early_stopping=True)
                    # predicted_texts = [self.tokenizer.decode(p, skip_special_tokens=True) for p in predictions]
                    # target_texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in batch['target_ids']]

                    # bleu_score = corpus_bleu([[t.split()] for t in target_texts], [p.split() for p in predicted_texts])
                    # rouge_score = self.rouge.get_scores(predicted_texts, target_texts, avg=True)

                    # wandb.log({"BLEU": bleu_score, "ROUGE-L": rouge_score["rouge-l"]["f"]})

                
                # save the model every 5000 iterations
                if iteration % 5000 == 0:
                    print(f'Saving model at epoch {epoch} and iteration {iteration}')
                    torch.save(self.model.state_dict(), f'trained_models/model_{epoch}_{iteration}.pth')
        
            torch.save(self.model.state_dict(), f'trained_models/model_complete.pth')

    def validate(self):
        self.model.eval()
        predictions = []
        actuals = []
        sources = []
        with torch.no_grad():
            for _, data in tqdm(enumerate(self.val_dataloader, 0)):
                source_ids = data["source_ids"].to("cpu", dtype=torch.long)
                source_mask = data["source_mask"].to("cpu", dtype=torch.long)
                target_ids = data["target_ids"].to("cpu", dtype=torch.long)

                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=source_ids, 
                    attention_mask=source_mask, 
                    max_length=150
                )

                preds = [
                    self.tokenizer.decode(
                        g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    for g in generated_ids
                ]
                target = [
                    self.tokenizer.decode(
                        t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    for t in target_ids
                ]
                source = [
                    self.tokenizer.decode(
                        s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    for s in source_ids
                ]


                sources.extend(source)
                predictions.extend(preds)
                actuals.extend(target)
        return sources, predictions, actuals
