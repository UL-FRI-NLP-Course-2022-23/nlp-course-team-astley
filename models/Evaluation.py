

def perplexity(transformer_model, test_data):
    sentences = test_data['endings']

    # compute perplexities via transformer model
    perplexities = transformer_model.calculate_perplexity(sentences)

    return perplexities


def negate_character_sentiment(character_sentiment):
    character, sentiment = character_sentiment
    return character, negate_sentiment(sentiment)


def negate_sentiment(sentiment):
    if sentiment is "negative":
        return "positive"
    elif sentiment is "positive":
        return "negative"
    else:
        return sentiment
