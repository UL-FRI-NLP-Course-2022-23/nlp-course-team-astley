from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
import torch
import os
import random
import json

from .common import to_encode_string
from .constants import *

# class for our stories dataset, which is used by pytorch for training
class StoryDataset(Dataset):
    def __init__(self, folder_path, char_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []
        file_names = os.listdir(folder_path)

        char_sent = {}
        with open(char_path, 'r') as file_characters:
            data = json.load(file_characters)
            for item in data:
                story_id = item["story_id"]
                characters = item["characters"]
                char_sent[story_id] = [f"{name} {attribute}" for name, attribute in characters.items()]

        idx = 0
        for file_name in file_names:
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r') as file:
                file_contents = file.read()
                words = file_contents.split()

                for i in range(0, len(words), INPUT_WORDS):
                    sentence = ' '.join(words[i:i+INPUT_WORDS])
                    # self.data.append((str(idx), sentence))
                    self.data.append((char_sent[file_name], sentence))
            idx += 1
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context, sentence = self.data[index]
        input = to_encode_string(context, sentence)

        encodings_dict = self.tokenizer(input,
                                        truncation=True,
                                        max_length=MAX_LEN,
                                        padding="max_length"
                                        )

        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        # Shifting the inputs and labels to align them happens inside the model,
        # so the labels are copies of the inputs.
        return {'label': torch.tensor(input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}
    

def load_story_dataset(path, char_path, tokenizer):
    
    dataset = StoryDataset(path, char_path, tokenizer)

    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    print(f'There are {train_size :,} samples for training, and {val_size :,} samples for validation testing')

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # train_dataloader = DataLoader(
    #     train_dataset,  # The training samples.
    #     sampler=RandomSampler(train_dataset),  # Select batches randomly
    #     batch_size=BATCH_SIZE  # Trains with this batch size.
    # )

    # # For validation the order doesn't matter, so we'll just read them sequentially.
    # validation_dataloader = DataLoader(
    #     val_dataset,  # The validation samples.
    #     sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
    #     batch_size=BATCH_SIZE  # Evaluate with this batch size.
    # )
    # return (train_dataloader, validation_dataloader) 
    
    return (train_dataset, val_dataset) 