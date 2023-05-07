import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline, DistilBertTokenizer, DistilBertForSequenceClassification
import json
import os
import json
import torch

# Download the NLTK package required for sentence tokenization
nltk.download('punkt')

class SentimentAnalysis:
    # Function to detect character names in the given story
    def detect_characters(self, story):
        # Load the pre-trained SpaCy model
        nlp = spacy.load("en_core_web_trf")
        # Process the story text to identify entities (such as character names)
        doc = nlp(story)
        characters = set()

        # Iterate through the entities found in the story text
        for ent in doc.ents:
            # If the entity is a person, add it to the set of characters
            if ent.label_ == "PERSON":
                characters.add(ent.text)

        # Return the list of unique character names
        return list(characters)

    # Function to extract sentences for each character in the story
    def extract_character_sentences(self, story, characters, window_size=4):
        # Tokenize the story into sentences using the NLTK sentence tokenizer
        sentences = sent_tokenize(story)
        # Initialize a dictionary to store sentences for each character
        character_sentences = {name: [] for name in characters}

        # Iterate through the sentences in the story
        for i, sentence in enumerate(sentences):
            # Check if a character's name appears in the sentence
            for name in characters:
                if re.search(r'\b' + re.escape(name) + r'\b', sentence, re.IGNORECASE):
                    # If so, get the surrounding context of the sentence (window size)
                    start = max(0, i - window_size + 1)
                    end = min(i + window_size, len(sentences))
                    context = ' '.join(sentences[start:end])
                    # Add the sentence and its context to the character's list of sentences
                    character_sentences[name].append(context)

        # Return the dictionary of character sentences
        return character_sentences

    # Function to perform sentiment analysis on the extracted sentences
    def sentiment_analysis(self, character_sentences, max_length=512):
        # Load the pre-trained sentiment analysis pipeline from the transformers library
        sentiment_analyzer = pipeline("sentiment-analysis")

        # Helper function to split text into chunks with a maximum length
        def split_into_chunks(text, max_length):
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0

            # Iterate through the words in the text
            for word in words:
                # Check if adding the word to the current chunk would exceed the maximum length
                if current_length + len(word) + 1 <= max_length:
                    # If not, add the word to the current chunk
                    current_chunk.append(word)
                    current_length += len(word) + 1
                else:
                    # Otherwise, start a new chunk with the current word
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word) + 1

            # Add the last chunk to the list of chunks
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks

        # Initialize a dictionary to store the sentiment scores for each character
        character_sentiments = {}
        # Analyze the sentiment of each character's extracted sentences
        for name, contexts in character_sentences.items():
            if contexts:
                # Split the contexts into chunks to ensure they don't exceed the maximum length allowed
                all_chunks = [split_into_chunks(context, max_length) for context in contexts]
                # Flatten the list of lists of chunks into a single list of chunks
                flattened_chunks = [chunk for chunks in all_chunks for chunk in chunks]
                # Get sentiment scores for each chunk using the pre-trained sentiment analysis pipeline
                sentiment_scores = sentiment_analyzer(flattened_chunks)
                # Calculate the compound sentiment score as the average of the individual sentiment scores
                compound_score = sum([score['score'] if score['label'] == 'POSITIVE' else -score['score'] for score in sentiment_scores]) / len(sentiment_scores)
            else:
                compound_score = 0.0

            # Store the character's sentiment score in the dictionary
            character_sentiments[name] = compound_score

        # Return the dictionary of character sentiment scores
        return character_sentiments

    def fine_tuned_sentiment_analysis(self, character_sentences, max_length=512):
        model_path = './models/sent_model/sentiment_model'
        tokenizer = DistilBertTokenizer.from_pretrained(model_path, local_files_only=True)
        model = DistilBertForSequenceClassification.from_pretrained(model_path, num_labels=2)

        # Helper function to split text into chunks with a maximum length
        def split_into_chunks(text, max_length):
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0

            # Iterate through the words in the text
            for word in words:
                # Check if adding the word to the current chunk would exceed the maximum length
                if current_length + len(word) + 1 <= max_length:
                    # If not, add the word to the current chunk
                    current_chunk.append(word)
                    current_length += len(word) + 1
                else:
                    # Otherwise, start a new chunk with the current word
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word) + 1

            # Add the last chunk to the list of chunks
            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks

        # Initialize a dictionary to store the sentiment scores for each character
        character_sentiments = {}
        # Analyze the sentiment of each character's extracted sentences
        for name, contexts in character_sentences.items():
            if contexts:
                inputs = tokenizer(' '.join(contexts), padding=True, truncation=True, return_tensors='pt')
                with torch.no_grad():
                    # inputs = inputs.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                    outputs = model(**inputs)
                    logits = outputs.logits

                predicted_label = torch.argmax(logits, dim=1).item()
                compound_score = "positive" if predicted_label == 1 else "negative"
            else:
                compound_score = "None"

            # Store the character's sentiment score in the dictionary
            character_sentiments[name] = compound_score

        # Return the dictionary of character sentiment scores
        return character_sentiments

    # Function to read stories from a directory and perform sentiment analysis on their characters
    def read_stories_get_sentiment(self, story_dir):
        stories = []
        file_names = os.listdir(story_dir)
        for file_name in file_names:
            file_path = os.path.join(story_dir, file_name)
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                story = file.read()

            # Detect the characters in the story
            characters = self.detect_characters(story)
            # Extract sentences and their context for each character
            character_sentences = self.extract_character_sentences(story, characters)
            # Perform sentiment analysis on the extracted sentences
            character_sentiments = self.sentiment_analysis(character_sentences)

            characters_info = {}
            characters_info["story_id"] = file_name
            characters_info["characters"] = {name: sentiment for name, sentiment in character_sentiments.items()}
            stories.append(characters_info)

        # Save the results to a JSON file
        with open("./results/stories_context.json", "w") as outfile:
            json.dump(stories, outfile, indent=4)

    def read_stories_get_sentiment_json(self, story_dir):
        stories = []
        with open('./results/stories_context.json', 'r+') as file:
            data = json.load(file)
            for story in data:
                file_path = os.path.join(story_dir, story["story_id"])
                with open(file_path, 'r', encoding='ISO-8859-1') as file:
                    story_text = file.read()

                # Detect the characters in the story
                characters = list(story["characters"].keys())
                # Extract sentences and their context for each character
                character_sentences = self.extract_character_sentences(story_text, characters)
                # Perform sentiment analysis on the extracted sentences
                character_sentiments = self.fine_tuned_sentiment_analysis(character_sentences)

                characters_info = {}
                characters_info["story_id"] = story["story_id"]
                characters_info["characters"] = {name: sentiment for name, sentiment in character_sentiments.items()}
                print(characters_info)
                stories.append(characters_info)

        # Save the results to a JSON file
        with open("./results/stories_context_new.json", "w") as outfile:
            json.dump(stories, outfile, indent=4)
