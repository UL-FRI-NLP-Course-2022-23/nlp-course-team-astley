from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, AdamW
from typing import List
import torch

from .StoryDataset import load_story_dataset
from .utils import to_encode_string
from .constants import *

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
            model_type = "gpt2"
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_type)
            self.model = GPT2LMHeadModel.from_pretrained(model_type)
            num_tokens = self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
            print("Special tokens added:", num_tokens)

            # Resize_token_embeddings to set the new vocabulary size
            self.model.resize_token_embeddings(len(self.tokenizer))
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained(load_path)
            self.model = GPT2LMHeadModel.from_pretrained(load_path)

        assert self.tokenizer.sep_token == "[SEP]"

        # Set GPU if available.
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)


    def generate_story(self):

        self.model.eval()
        sample_outputs = []
        final_outputs = []
        # Set up input prompt and control string
        for i in range(4):
            control_string = str(i) 
            input_prompt = """As Fawkes approached the tree trunk, Duke crept closer, ready to pounce if Fawkes made any sudden movements."""
            
            prompt = to_encode_string(control_string, input_prompt)

            generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
            generated = generated.to(self.device)

            # Generate text using the model
            output = self.model.generate(
                input_ids=generated,
                max_length=500,
                repetition_penalty=1.2,
                num_return_sequences=1
            )

            sample_outputs.append(output[0])

            # sample_outputs = model.generate(generated, 
            #                                 do_sample=True,   
            #                                 min_length=50, 
            #                                 max_length=500,
            #                                 top_k=30,                                 
            #                                 top_p=0.7,        
            #                                 temperature=0.9,
            #                                 repetition_penalty=2.0,
            #                                 num_return_sequences=10
            #                                 )

            for i, sample_output in enumerate(sample_outputs):
                text = self.tokenizer.decode(sample_output, skip_special_tokens=False)
                # a = len(title) + len(','.join(keywords))    
                final_outputs.append(text)
                with open("./results/story"+str(i)+".txt", "w", encoding="utf-8") as f:
                    f.write(text)
        
        return sample_outputs


    def train(self, path, output_path):

        train_dataset, val_dataset = load_story_dataset(path, self.tokenizer)


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
            fp16=True,
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

