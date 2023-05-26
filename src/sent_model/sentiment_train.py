import json
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch

model_name = 'distilbert-base-uncased-finetuned-sst-2-english'

# Step 1: Load the dataset from JSON file
def load_dataset(file_path):
    sentences = []
    scores = []

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        for sample in data:
            sentences.append(sample['sentence'])
            scores.append([sample['score'], sample['trust']])

    return sentences, scores

# Replace 'data.json' with your JSON file path
sentences, scores = load_dataset('output_updated.json')

# Step 2: Split the dataset into training and testing sets
# Replace this step with your preferred method of splitting the data

# Step 3: Tokenize the sentences
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
encoded_inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Step 4: Convert scores to torch tensors
def to_dataset_label(vector):
    arr = [0, 0]

    if vector[0] > 0:
        arr[0] = 1
    elif vector[0] < 0:
        arr[0] = -1

    # arr[1] = min(0.25 * vector[1], 1.0)

    return arr[0]

scores = list(map(to_dataset_label, scores))
labels = torch.tensor(scores)

# Step 5: Fine-tune the pre-trained model
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=3)
optimizer = AdamW(model.parameters(), lr=1e-5)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.train()
model.to(device)

num_epochs = 5
batch_size = 16

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, len(sentences), batch_size):
        inputs = {
            'input_ids': encoded_inputs['input_ids'][i:i+batch_size].to(device),
            'attention_mask': encoded_inputs['attention_mask'][i:i+batch_size].to(device),
            'labels': labels[i:i+batch_size].to(device)
        }

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / (len(sentences) // batch_size)
    print(f'Epoch {epoch+1} - Average Loss: {average_loss:.4f}')

# Step 6: Save the trained model
model.save_pretrained('sentiment_model')
tokenizer.save_pretrained('sentiment_model')
