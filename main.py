import torch
from models import StoryEndingGenerator

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
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
        self.ner = BertForTokenClassification.from_pretrained('bert-base-uncased')
        # self.tokenizer = BertTokenizer.from_pretrained(model_name)
        # self.model = BertForSequenceClassification.from_pretrained(model_name)

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.model(**inputs)
        return torch.argmax(outputs.logits, dim=1).item()

    def predict_named_entities(self, text):
        inputs = self.tokenizer(text, return_tensors='pt')
        outputs = self.ner(**inputs)
        return outputs.logits


# Character Sentiment Reversal
def reverse_sentiment(sentiment):
    if sentiment == 0:
        return 2
    elif sentiment == 2:
        return 0
    return 1


def read_file_as_string(filename):
    with open(filename, 'r') as file:
        return file.read()


def sentence_extraction(text):
    tmp = text.replace('.', '<SE>')
    tmp = tmp.replace('?', '<SE>')
    tmp = tmp.replace('!', '<SE>')
    return tmp.split('<SE>')


def extract_sentiment(sa, sentences):
    sentiment_matrix = {}

    for sentence in sentences:
        # sentiment from sentence
        sentiment = sentiment_analysis.predict_sentiment(sentence)

        # TODO: get named entities from sentence
        # print(sentiment_analysis.predict_named_entities(sentence))
        entities = ["Michael", "Rachel"]

        # add to resulting array (dict)
            # entiments per person
            # sentiments per pairs of persons
        for e1 in entities:
            for e2 in entities:
                if e1 is e2:
                    if e1 not in sentiment_matrix:
                        sentiment_matrix[e1] = [0, 0, 0]
                    sentiment_matrix[e1][sentiment] += 1
                else:
                    if (e1, e2) not in sentiment_matrix:
                        sentiment_matrix[(e1, e2)] = [0, 0, 0]
                    sentiment_matrix[(e1, e2)][sentiment] += 1

    # return matrix
    return sentiment_matrix


# Main
if __name__ == '__main__':
    # dataset = load_dataset('./zgodbice/1.txt')

    # Train the sentiment analysis model
    # train_sentiment_model(None)
    sentiment_analysis = SentimentAnalysis()

    # read a file
    # TODO: story = read_file_as_string('./zgodbice/1.txt')
    story = "Michael likes Rachel. Michael is a bad man. Rachel is great. Test"
    # separate to sentences
    sentences = sentence_extraction(story)

    data = extract_sentiment(sentiment_analysis, sentences)
    # display data
    print(data)

    # print(sentiment_analysis.predict_sentiment(["I like you.", "I love you."]))
    # print(sentiment_analysis.do_something(["Michael ate an apple. Michael is fat."]))
    # -------------------------------------

    
    # basic gpt2
    ending_model = StoryEndingGenerator()
    
    # fine-tunned gpt2
    # ending_model = StoryEndingGenerator("./content")
    
    # @param: path to dataset
    # @param: output path of model while training
    ending_model.train("../archive/dataset", "test")

    # # Reverse sentiment to see how ending will change
    # for story in generated_stories:
    #     sentiment = sentiment_analysis.predict_sentiment(story)
    #     reversed_sentiment = reverse_sentiment(sentiment)
    #     print(f"Story: {story}")
    #     print(f"Original Sentiment: {sentiment}")
    #     print(f"Reversed Sentiment: {reversed_sentiment}")
    #     print()
    ending_model.generate_story()
