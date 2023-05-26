import csv
import json

import os
from models import StoryEndingGenerator, SentimentAnalysis, to_encode_string, perplexity
from models.Evaluation import negate_sentiment

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"



# Main
if __name__ == '__main__':
    # dataset = load_dataset('./zgodbice/1.txt')

    # Train the sentiment analysis model
    # train_sentiment_model(None)
    # sentiment_analysis = SentimentAnalysis()
    # sentiment_analysis.read_stories_get_sentiment_json("./fairy_tales")

    # basic gpt2
    # ending_model = StoryEndingGenerator("content")
    # ending_model = StoryEndingGenerator("../test")
    ending_model = StoryEndingGenerator("NLP_last_model")

    # fine-tunned gpt2
    # ending_model = StoryEndingGenerator("./content")

    # @param: path to dataset
    # @param: output path of model while training
    # ending_model.train("./ROC_dataset/dataset_ROC.csv", "test", './results/stories_context_new.json', short=False)

    #context = ["Tom negative"]
    #story = """Tom had a very short temper. One day a guest made him very angry. He punched a hole in the wall of his house. Tom's guest became afraid and left quickly."""
    # Tom sat on his couch filled with regret about his actions.
    #prompt = to_encode_string(context, story)

    # @param: prompt in form of "context [SEP] story", use to_encode_string
    # @param: skip special tokens when decoding the string
    #outputs = ending_model.generate_story(prompt, skip_special=False)
    #print(outputs)

    # evaluation


    num_comparisons = 0
    num_equal = 0
    num_equal_non_none = 0
    results = []

    # get character sentiment
    with open("ROC_dataset/test_dataset_sentiment.json", 'r') as file_sentiment:
        character_sentiment = json.load(file_sentiment)
        character_sentiment.sort(key=lambda x: x['story_id'])

        with open('ROC_dataset/test.csv', 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            header = next(reader)
            for i, row in enumerate(reader):
                print(i)


                # extract story
                story = " ".join(row[0:4])
                endings = row[4:]

                # negate sentiment
                characters = list(character_sentiment[i]["characters"].keys())
                # characters = character_sentiment[i]["characters"]
                character_sentiments = {c: character_sentiment[i]["characters"][c] for c in characters}
                negated_character_sentiments = {c: negate_sentiment(character_sentiment[i]["characters"][c]) for c in characters}

                # generate endings
                # context = ["Duke negative, Fawkes positive"]
                context = [", ".join(" ".join([c, negated_character_sentiments[c]]) for c in characters)]
                # print(context)
                # print(story)
                # story = """Tom had a very short temper. One day a guest made him very angry. He punched a hole in the wall of his house. Tom's guest became afraid and left quickly."""
                # Tom sat on his couch filled with regret about his actions.
                prompt = to_encode_string(context, story)
                outputs = ending_model.generate_story(prompt, skip_special=False)

                # generate perplexity
                perp_score = perplexity(ending_model, outputs)

                # compare sentiment
                sentiment_analysis = SentimentAnalysis("sentiment_model")
                # base_sentiment = sentiment_analysis.extract_character_sentences(outputs["prompt"].split(" [SEP] ")[1],
                #                                                                   characters)
                # sentiments = [sentiment_analysis.extract_character_sentences(ending, characters) for ending in
                #               outputs["endings"].values()]
                base_char_ss = [sentiment_analysis.extract_character_sentences(ending, characters) for ending in endings]
                # base_sentiment = [sentiment_analysis.fine_tuned_sentiment_analysis(ss) for ss in base_char_ss]
                base_sentiment = [sentiment_analysis.fine_tuned_sentiment_analysis(ss) for ss in base_char_ss]
                sentiments = [sentiment_analysis.fine_tuned_sentiment_analysis(
                    sentiment_analysis.extract_character_sentences(ending, characters)) for ending in
                    outputs["endings"].values()]

                # print(base_sentiment)
                # [{'Bob': 'None'}, {'Bob': 'None'}]
                # print(sentiments)
                # [{'Bob': 'positive'}, {'Bob': 'positive'}, {'Bob': 'None'}, {'Bob': 'None'}, {'Bob': 'negative'}]

                # collect results
                # original sentiments
                story_results = [character_sentiments]

                # sentiments based on ending pairs
                for base_ending_sentiment in base_sentiment:
                    ending_list = []
                    for idx, ending_sentiment in enumerate(sentiments):
                        ressss = {}
                        for character in characters:
                            ressss[character] = (base_ending_sentiment[character], ending_sentiment[character])
                            num_comparisons += 1
                            if base_ending_sentiment[character] is negate_sentiment(ending_sentiment[character]):
                                num_equal += 1
                                if base_ending_sentiment[character] is not "None":
                                    num_equal_non_none += 1
                        ending_list.append((ressss, perp_score[idx]))

                    # generated ending sentiments and perplexities
                    story_results.append(ending_list)

                results.append(story_results)
                # break

    print("RESULTS")
    print(num_comparisons)
    print(num_equal)
    print(num_equal_non_none)
    print(num_equal / num_comparisons)
    print((num_equal - num_equal_non_none) / num_comparisons)

    print(results)
