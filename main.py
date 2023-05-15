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
    ending_model = StoryEndingGenerator("content")
    
    # fine-tunned gpt2
    # ending_model = StoryEndingGenerator("./content")
    
    # @param: path to dataset
    # @param: output path of model while training
    # ending_model.train("./fairy_tales", "test", './results/stories_context_new.json')

    context = ["Duke negative, Fawkes positive"]
    story = """They quickly rigged up a rude sled, made of poles, put the head of Toover Hek on it, and drew it off to the knight's castle. There it was exposed, on a sharpened stake of wood, in front of the gate. For a whole week it was the sport of the community, and the lads and maidens danced and sang and all the people rejoiced. After the ogre’s head was taken down, it was set in the ground at the side of a brook, and used for women to stand or kneel on, while washing clothes. In time it was polished as ivory and shone in the sun. As for Heinrich, he hitched up four yoke of oxen, and tying an iron chain around the fir tree trunk, which formed the giant’s club, he dragged it to his barnyard and there had it chopped up. It made a load of firewood which lasted him all winter. Now that the roads were safe for all travelers, Heinrich and Grietje, and the knight, in thankfulness to the Holy Virgin fixed a pretty little shrine to one of the forest trees."""
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
