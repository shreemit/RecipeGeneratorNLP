import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import DataLoader
from train import RecipeDataset, T5Trainer
from sklearn.model_selection import train_test_split
import pandas as pd
import wandb

def main():
    # Load data
    train_df = pd.read_csv('train_df.csv', index_col=0)
    train_df, val_df = train_test_split(train_df, test_size=0.1)
    
    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    device = 'mps'

    # Initialize tokenizer and model
    tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
    model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small').to(device)

    # Initialize datasets
    train_dataset = RecipeDataset(train_df, tokenizer, 40, 128)
    val_dataset = RecipeDataset(val_df[0:1000], tokenizer, 40, 128)

    # Initialize dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    wandb.init(project="T5-training", entity="shreemit27")
    wandb.watch(model, log="all")

    # Initialize trainer
    trainer = T5Trainer(model, tokenizer, train_dataloader, val_dataloader, device, epochs = 1)

    # Train model
    trainer.train()

    # Validate model
    model.to('cpu')
    sources, predictions, actuals = trainer.validate()

    # Print some results
    for source, prediction, actual in zip(sources[:10], predictions[:10], actuals[:10]):
        print(f'Source: {source}\nPrediction: {prediction}\nActual: {actual}\n')




if __name__ == '__main__':
    main()