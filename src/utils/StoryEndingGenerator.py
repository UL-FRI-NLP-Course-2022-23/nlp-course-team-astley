import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, AdamW
from typing import List
import torch

from .StoryDataset import load_story_dataset
from .ROCStoryDataset import load_ROC_dataset
from .common import to_encode_string
from .constants import *


def get_first_sentence(ending):
    sentences = ending.split('\n')
    first_sentence = sentences[0]
    cleaned_sentence = ' '.join(word for word in first_sentence.split(' ') if word != '[PAD]')
    return cleaned_sentence


# adding special tokens
#  https://github.com/huggingface/transformers/issues/17690
# how labels are formed
# https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt 
# GPT-2 Fine-Tuning w/ Hugging Face & PyTorch.ipynb
# https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=GLs72DuMODJO
# Conditional Text Generation with GPT-2.ipynb
# https://colab.research.google.com/drive/1vnpMoZoenRrWeaxMyfYK4DDbtlBu-M8V?usp=sharing#scrollTo=QILzrXuoRhaF


class StoryEndingGenerator:
    def __init__(self, load_path=None):
        
        if (load_path == None):
            # load path is not given, load pretrained GPT2 from HF
            model_type = "gpt2"
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
            self.model = GPT2LMHeadModel.from_pretrained(model_type)
            num_tokens = self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
            print("Special tokens added:", num_tokens)

            # Resize_token_embeddings to set the new vocabulary size
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            # load path is given, load our GPT2
            self.tokenizer = GPT2Tokenizer.from_pretrained(load_path)
            self.model = GPT2LMHeadModel.from_pretrained(load_path)

        # Sanity check if the added token was really added
        assert self.tokenizer.sep_token == "[SEP]"

        # Set GPU if available
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)


    # generate 5 versions of story, write them to /results and 
    # also return them as an array.
    def generate_story(self, prompt, skip_special = True):
        self.model.eval()
        
        outputs = {
            "prompt": prompt,
            "endings": {}
        } 
        prompt_len = len(prompt)
  
        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(self.device)
        attention_mask = (generated != self.tokenizer.pad_token_id)

        sample_outputs = self.model.generate(generated,
                                        attention_mask=attention_mask,
                                        do_sample=True,   
                                        min_length=50, 
                                        max_length=100,
                                        top_k=20,                                 
                                        top_p=0.8,        
                                        temperature=0.5,
                                        repetition_penalty=2.0,
                                        num_return_sequences=5
                                        )
        
        for i, sample_output in enumerate(sample_outputs):
            text = self.tokenizer.decode(sample_output, skip_special_tokens=skip_special)
            outputs["endings"][i] = get_first_sentence(text[prompt_len:])
        
        return outputs


    def train(self, path, output_path, char_path, short=False):
        if(short):
            train_dataset, val_dataset = load_story_dataset(path, char_path, self.tokenizer)
        else:
            train_dataset, val_dataset = load_ROC_dataset(path, char_path, self.tokenizer)

        # Freeze all layers except the last TRAINABLE_LAYERS
        for i, m in enumerate(self.model.transformer.h):
            if i < 11-TRAINABLE_LAYERS:
                for parameter in m.parameters():
                    parameter.requires_grad = False

        for parameter in self.model.transformer.ln_f.parameters():
            parameter.requires_grad = False

        for parameter in self.model.lm_head.parameters():
            parameter.requires_grad = False

        # Set training arguments and start training
        training_args = TrainingArguments(
            output_dir=output_path,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            evaluation_strategy="epoch",
            logging_strategy="epoch",
            save_strategy="epoch",
            gradient_accumulation_steps=BATCH_UPDATE,
            #fp16=True,
            # fp16_opt_level=APEX_OPT_LEVEL,
            warmup_steps=WARMUP_STEPS,
            learning_rate=LR,
            adam_epsilon=EPS,
            weight_decay=0.01,
            save_total_limit=1,
            load_best_model_at_end=True,
        )

        # optimizer = AdamW(model.parameters(), lr=LR, eps=EPS)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # optimizer=optimizer,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        trainer.save_model()

    def calculate_perplexity(self, endings):
        self.model.eval()

        perplexities = []
        with torch.no_grad():
            for i, ending in endings.items():
                if ending is "":
                    perplexities.append(None)
                    continue
                encoded_input = self.tokenizer.encode(ending, return_tensors="pt").to(self.device)
                output = self.model(encoded_input, labels=encoded_input)
                loss = output.loss
                perplexity = torch.exp(loss)
                perplexities.append(np.log(perplexity.item()))

        return perplexities
