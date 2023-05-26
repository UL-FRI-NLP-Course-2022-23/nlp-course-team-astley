from torch.utils.data import Dataset, random_split, DataLoader, RandomSampler, SequentialSampler
import torch
import os
import random
import json
import csv

from .common import to_encode_string
from .constants import *

# class for our stories dataset, which is used by pytorch for training
class ROCStoryDataset(Dataset):
    def __init__(self, file_path, char_path, tokenizer):
        self.tokenizer = tokenizer
        self.data = []

        char_sent = {}
        with open(char_path, 'r') as file_characters:
            data = json.load(file_characters)
            for item in data:
                story_id = item["story_id"]
                characters = item["characters"]
                char_sent[story_id] = [f"{name} {attribute}" for name, attribute in characters.items()]

        with open(file_path, 'r') as file:

            reader = csv.reader(file)
            header = next(reader)
            idx = 0

            # Iterate over each row in the CSV file
            for row in reader:
                # Access the sentences in each row
                sentence = " ".join(row[:4])
                ending = row[4]
                self.data.append((["test", "positive"], sentence, ending)) #TODO: zamenjaj "test" z char_sent[idx]
                idx += 1
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context, sentence, ending = self.data[index]
        input = to_encode_string(context, sentence)
        
        # TODO: trenutno imamo samo ending kot label, probaj se z sentence + label
        # TODO: probaj se s tem da sta sentence in ending (label in input) ista z endingom
        # ending = sentence + " " + ending

        encodings_dict = self.tokenizer(input,
                                        truncation=True,
                                        max_length=MAX_LEN,
                                        padding="max_length"
                                        )
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']
        
        
        ending_encodings_dict = self.tokenizer(ending,
                                        truncation=True,
                                        max_length=MAX_LEN,
                                        padding="max_length"
                                        )

        ending_input_ids = ending_encodings_dict['input_ids']

        return {'label': torch.tensor(ending_input_ids),
                'input_ids': torch.tensor(input_ids),
                'attention_mask': torch.tensor(attention_mask)}
    

def load_ROC_dataset(path, char_path, tokenizer):
    
    dataset = ROCStoryDataset(path, char_path, tokenizer)

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