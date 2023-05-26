from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

model_path = './sentiment_model'
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Prepare the input sentence
sentence = "He is a evil, but in love."

# Tokenize the sentence
inputs = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')

# Perform the prediction
with torch.no_grad():
    # inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    outputs = model(**inputs)
    logits = outputs.logits
    print(outputs, logits)

# Get the predicted label
predicted_label = torch.argmax(logits, dim=1).item()

# Convert the predicted label to sentiment (0 for negative, 1 for positive)
sentiment = "positive" if predicted_label == 1 else "negative"

# Print the result
print(f"The sentiment of the sentence '{sentence}' is {sentiment}.")
