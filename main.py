import torch
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
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model('./sentiment_model')

# Fine-tune GPT-2 for story generation
def train_story_model(dataset):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    texts = [text for text, sentiment in dataset]
    text_dataset = TextDataset(tokenizer=tokenizer, file_path=None, text=texts, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./story_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=100,
        save_steps=500,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=text_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model('./story_model')

# Sentiment Analysis
class SentimentAnalysis:
    def __init__(self, model_name='./sentiment_model'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name)

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

# Story Generation
class StoryEndingGeneration:
    def __init__(self, model_name='./story_model'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def generate_ending(self, prompt, max_length=300, num_return_sequences=1):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Main
if __name__ == '__main__':
    dataset = load_dataset('./path/to/your/dataset.txt')

    # Train the sentiment analysis model
    train_sentiment_model(dataset)

    # Train the story generation model
    train_story_model(dataset)

    # Load the trained models
    sentiment_analysis = SentimentAnalysis()
    story_ending_generation = StoryEndingGeneration()

    # Generate a story
    prompt = "Bla bla bla"
    generated_stories = story_ending_generation.generate_ending(prompt, max_length=100, num_return_sequences=5)

    # Reverse sentiment to see how ending will change
    for story in generated_stories:
        sentiment = sentiment_analysis.predict_sentiment(story)
        reversed_sentiment = reverse_sentiment(sentiment)
        print(f"Story: {story}")
        print(f"Original Sentiment: {sentiment}")
        print(f"Reversed Sentiment: {reversed_sentiment}")
        print()
