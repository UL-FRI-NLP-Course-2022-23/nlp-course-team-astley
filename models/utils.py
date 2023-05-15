from .constants import *
import random

# used when concatenating characters sentiments
def join_keywords(keywords, randomize=True):

    # random sampling and shuffle
    if randomize:
        random.shuffle(keywords)

    return ', '.join(keywords)

def to_encode_string(context, sentence):
    keywords = join_keywords(context)
    prompt = keywords + " " + SPECIAL_TOKENS["sep_token"] + " " + sentence
    return prompt
