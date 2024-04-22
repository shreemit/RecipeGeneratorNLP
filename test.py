import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from train import RecipeDataset
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import bert_score

class TestModel:
    def __init__(self, model_path, test_data_path, tokenizer, source_len, summ_len, batch_size=16):
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = tokenizer
        self.tokenizer.add_special_tokens({'pad_token': '<PAD>', 'additional_special_tokens': ['<I>']})
        self.model.resize_token_embeddings(len(tokenizer))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.test_data = pd.read_csv(test_data_path)
        self.dataset = RecipeDataset(self.test_data, self.tokenizer, source_len, summ_len)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size)

    def test(self):
        self.model.eval()
        predictions = []
        sources = []
        with torch.no_grad():
            for _, data in enumerate(self.dataloader, 0):
                source_ids = data["source_ids"].to(self.device, dtype=torch.long)
                source_mask = data["source_mask"].to(self.device, dtype=torch.long)

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
                source = [
                    self.tokenizer.decode(
                        s, skip_special_tokens=True, clean_up_tokenization_spaces=True
                    )
                    for s in source_ids
                ]

                sources.extend(source)
                predictions.extend(preds)

                # calculate BLEU score
                bleu_score = corpus_bleu(predictions, sources)
                print(f"BLEU Score: {bleu_score}")

                # calculate ROUGE score
                rouge = Rouge()
                rouge_score = rouge.get_scores(predictions, sources, avg=True)

                # calculate BERT score
                P, R, F1 = bert_score.score(preds, source, lang='en', verbose=True)
                print(f"BERT Score: {F1.mean()}")
            
        return sources, predictions

if __name__ == "__main__":
    model_path = "model_0_5000.pth"
    test_data_path = "test_df.csv"
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    source_len = 512
    summ_len = 150

    tester = TestModel(model_path, test_data_path, tokenizer, source_len, summ_len)
    sources, predictions = tester.test()

    for source, prediction in zip(sources[0:10], predictions[0:10]):
        print(f"Source: {source}")
        print(f"Prediction: {prediction}")
        print("\n")