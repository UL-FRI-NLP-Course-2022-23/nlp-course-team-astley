import torch
import os
from models import StoryEndingGenerator, SentimentAnalysis, to_encode_string, perplexity



os.environ["KMP_DUPLICATE_LIB_OK"] = "True"



# Main
if __name__ == '__main__':
    # dataset = load_dataset('./zgodbice/1.txt')

    # Train the sentiment analysis model
    # train_sentiment_model(None)
    # sentiment_analysis = SentimentAnalysis()
    # sentiment_analysis.read_stories_get_sentiment_json("./fairy_tales")
    
    # basic gpt2
    # ending_model = StoryEndingGenerator("content")
    ending_model = StoryEndingGenerator("../test")
    
    # fine-tunned gpt2
    # ending_model = StoryEndingGenerator("./content")
    
    # @param: path to dataset
    # @param: output path of model while training
    # ending_model.train("./fairy_tales", "test", './results/stories_context_new.json')

    context = ["Mirko negative"]
    story = """Well, because, said Mirko, I'm anxious to go back to my country, but I am also extremely sorry to leave behind this sumptuous diamond castle, six stories high, which belonged to your father, for there is nothing like it in my country. Well, my love, said the princess, don't trouble about that. I will transform the castle into a golden apple at once, and sit in the middle of it, and all you will have to do is to put the apple into your pocket, and then you can take me with you and the castle too, and when you arrive at home you can re-transform me wherever you like. Thereupon the pretty princess jumped down from her horse, handed the reins to Mirko, took out a diamond rod, and commenced to walk round the diamond castle, gently beating the sides of it with the diamond rod, and the castle began to shrink and shrunk as small as a sentry box, and then the princess jumped inside of it, and the whole shrivelled up into a golden apple, the diamond rod lying by the side of it. Prince Mirko picked up the golden apple and the diamond rod, and put them into his pocket, and then got on horseback, and, taking Doghead's horse by the bridle, he rode quietly home. Having arrived at home, Mirko had the horses put in the stables, and then walked into the royal palace, where he found the old king and Knight Mezey quite content and enjoying themselves. He reported to them that he had conquered even Doghead, and that he had killed him; but the old king and Knight Mezey doubted his words. Therefore Prince Mirko took them both by their arms, and said to them, Come along with me, and you can satisfy yourselves, with your own eyes, that I have conquered Doghead, because I"""
    prompt = to_encode_string(context, story)
    
    # @param: prompt in form of "context [SEP] story", use to_encode_string
    # @param: skip special tokens when decoding the string
    outputs = ending_model.generate_story(prompt, skip_special=False)
    print(outputs)

    # evaluation

    # 1. perplexity
    perp_score = perplexity()
    print(perp_score)
    # 2. check if ending sentiment is opposite 
    # 3. self supervised?
