from constants import *

def to_encode_string(context, sentence):
    return context + " " + SPECIAL_TOKENS["sep_token"] + " " + sentence
