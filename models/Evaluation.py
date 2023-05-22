

def perplexity(transformer_model, test_data):
    sentences = test_data['endings']

    # compute perplexities via transformer model
    perplexities = transformer_model.calculate_perplexity(sentences)

    return perplexities
