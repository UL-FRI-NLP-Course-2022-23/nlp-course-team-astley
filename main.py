import torch
import os
from models import StoryEndingGenerator, SentimentAnalysis



os.environ["KMP_DUPLICATE_LIB_OK"] = "True"



# Main
if __name__ == '__main__':
    # dataset = load_dataset('./zgodbice/1.txt')

    # Train the sentiment analysis model
    # train_sentiment_model(None)
    # sentiment_analysis = SentimentAnalysis()
    # sentiment_analysis.read_stories_get_sentiment_json("./fairy_tales")
    
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
