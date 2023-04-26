import torch
from models import StoryEndingGenerator

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Load your dataset
def load_dataset(file_path):
    # We need to change, because I don't know in what form our data will be
    with open(file_path, 'r') as f:
        data = [line.strip().split('\t') for line in f]
    return [(text, int(sentiment)) for text, sentiment in data]

# Dataset class for sentiment analysis
class SentimentAnalysisDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, sentiment = self.data[idx]
        inputs = self.tokenizer(text, max_length=self.max_length, padding='max_length', truncation=True, return_tensors='pt')
        inputs['labels'] = torch.tensor(sentiment)
        return inputs

# Fine-tune BERT for sentiment analysis
def train_sentiment_model(dataset):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

    train_dataset = SentimentAnalysisDataset(dataset, tokenizer)

    training_args = TrainingArguments(
        output_dir='./sentiment_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=100,
        save_steps=500,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        # args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model('./sentiment_model')

# Sentiment Analysis
class SentimentAnalysis:
    def __init__(self, model_name='./sentiment_model'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # self.model = BertForSequenceClassification.from_pretrained(model_name)

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()

# Character Sentiment Reversal
def reverse_sentiment(sentiment):
    if sentiment == 0:
        return 2
    elif sentiment == 2:
        return 0
    return 1


# Main
if __name__ == '__main__':
    # dataset = load_dataset('.')

    # Train the sentiment analysis model
    # train_sentiment_model(dataset)
    # sentiment_analysis = SentimentAnalysis()
    # -------------------------------------

    
    # basic gpt2
    ending_model = StoryEndingGenerator()
    
    # fine-tunned gpt2
    # ending_model = StoryEndingGenerator("./content")
    
    # @param: path to dataset
    # @param: output path of model while training
    ending_model.train("../archive/dataset", "test")

    ending_model.generate_story()
