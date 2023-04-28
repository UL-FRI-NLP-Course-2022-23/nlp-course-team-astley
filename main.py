import torch
import os
from models import StoryEndingGenerator, SentimentAnalysis

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertForTokenClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# reads the entire file into one string
def read_file_as_string(filename):
    with open(filename, 'r') as file:
        return file.read()

# Roughly divides a given string into a list of sentences (strings)
# could be improved
def sentence_extraction(text):
    tmp = text.replace('.', '<SE>')
    tmp = tmp.replace('?', '<SE>')
    tmp = tmp.replace('!', '<SE>')
    return tmp.split('<SE>')


# Extracts sentiment from a list of sentences based on SentimentAnalysis sa
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
    sentiment_analysis.read_stories_get_sentiment("./fairy_tales")

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
    
    # basic gpt2
    ending_model = StoryEndingGenerator()
    
    # fine-tunned gpt2
    # ending_model = StoryEndingGenerator("./content")
    
    # @param: path to dataset
    # @param: output path of model while training
    ending_model.train("./fairy_tales", "test")

    # # Reverse sentiment to see how ending will change
    # for story in generated_stories:
    #     sentiment = sentiment_analysis.predict_sentiment(story)
    #     reversed_sentiment = reverse_sentiment(sentiment)
    #     print(f"Story: {story}")
    #     print(f"Original Sentiment: {sentiment}")
    #     print(f"Reversed Sentiment: {reversed_sentiment}")
    #     print()
    ending_model.generate_story()
