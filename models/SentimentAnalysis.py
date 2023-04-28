import re
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import json
import os

nltk.download('punkt')

class SentimentAnalysis:
    def detect_characters(self, story):
        nlp = spacy.load("en_core_web_trf")
        doc = nlp(story)
        characters = set()

        for ent in doc.ents:
            if ent.label_ == "PERSON":
                characters.add(ent.text)

        return list(characters)

    def extract_character_sentences(self, story, characters, window_size=4):
        sentences = sent_tokenize(story)
        character_sentences = {name: [] for name in characters}

        for i, sentence in enumerate(sentences):
            for name in characters:
                if re.search(r'\b' + re.escape(name) + r'\b', sentence, re.IGNORECASE):
                    start = max(0, i - window_size + 1)
                    end = min(i + window_size, len(sentences))
                    context = ' '.join(sentences[start:end])
                    character_sentences[name].append(context)

        return character_sentences

    def sentiment_analysis(self, character_sentences, max_length=512):
        sentiment_analyzer = pipeline("sentiment-analysis")

        def split_into_chunks(text, max_length):
            words = text.split()
            chunks = []
            current_chunk = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 <= max_length:
                    current_chunk.append(word)
                    current_length += len(word) + 1
                else:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = len(word) + 1

            if current_chunk:
                chunks.append(' '.join(current_chunk))

            return chunks

        character_sentiments = {}
        for name, contexts in character_sentences.items():
            if contexts:
                all_chunks = [split_into_chunks(context, max_length) for context in contexts]
                flattened_chunks = [chunk for chunks in all_chunks for chunk in chunks]
                sentiment_scores = sentiment_analyzer(flattened_chunks)
                compound_score = sum([score['score'] if score['label'] == 'POSITIVE' else -score['score'] for score in sentiment_scores]) / len(sentiment_scores)
            else:
                compound_score = 0.0

            character_sentiments[name] = compound_score

        return character_sentiments

    def read_stories_get_sentiment(self, story_dir):
        stories = []
        file_names = os.listdir(story_dir)
        for file_name in file_names:
            file_path = os.path.join(story_dir, file_name)
            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                story = file.read()

            characters = self.detect_characters(story)
            character_sentences = self.extract_character_sentences(story, characters)
            character_sentiments = self.sentiment_analysis(character_sentences)

            characters_info = {}
            characters_info["story_id"] = file_name
            characters_info["characters"] = {name: sentiment for name, sentiment in character_sentiments.items()}
            stories.append(characters_info)

        with open("./results/stories_context.json", "w") as outfile:
            json.dump(stories, outfile, indent=4)
