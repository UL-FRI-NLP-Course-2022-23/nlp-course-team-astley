import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification


class StoryEndingGeneration:
    def __init__(self, train=False, model_name=None):

        if(model_name == None):
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)

        if(train):
            pass# './story_model'

    def generate_ending(self, prompt, max_length=300, num_return_sequences=1):
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=num_return_sequences)
        return [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Fine-tune GPT-2 for story generation
def train_story_model(data_dir):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    texts = [text for text, sentiment in []]
    text_dataset = TextDataset(tokenizer=tokenizer, file_path=None, text=texts, block_size=128)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./story_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=100,
        save_steps=500,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=text_dataset,
        data_collator=data_collator,
    )

    # --------------------------------
    # Load the text files into a list
    texts = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.txt'):
            file_path = os.path.join(data_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                texts.append(text)

    # Create a TextDataset from the list of texts
    text_dataset = TextDataset(tokenizer=tokenizer, file_path=None, text=texts, block_size=128)

    # Set the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Set the training arguments
    training_args = TrainingArguments(
        output_dir='./story_model',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        logging_steps=100,
        save_steps=500,
        seed=42,
    )

    # Initialize the trainer and start training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=text_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model('./story_model')
