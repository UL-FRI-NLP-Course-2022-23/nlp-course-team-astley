# Natural language processing course 2022/23: `Literacy situation models knowledge base creation`

## How to run
For running our project, please open **run.ipynb** and see the instructions there. Note that if you don't have Jupyter installed, then you need to first run this command in the terminal: **pip install jupyter**, before opening Jupyter Notebook.

Here you can download our [Fine-tune GPT2](https://unilj-my.sharepoint.com/:f:/g/personal/lb4684_student_uni-lj_si/Er6Mr6tCjLBLvyNwu0h3PF4BU992FwGcOfERVh2uEhx9aA?e=KCT1lg)

Our Fine-tune Sentiment model is avaliable [here](https://drive.google.com/file/d/1AJUJJIqDjHqGwzm1zOueAJIE631bACat/view?usp=sharing)


## Structure of repository
- **REPORT**: contains the final report of our project.
- **datasets/**: includes all the datasets we used during the project.
    * **fairy_tales**: database of more than 1000 stories, which was abandoned due to the poor performance of our model.
    *  **ROC_dataset**: ROC Stories Corpus which was used for training of our GPT2 and evaluating stories with sentiment analysis.
    * **sentiment_dataset**: dataset which contains labelled scores of sentences and predictions of our sentiment model, can be used in self-supervised learning fashion to further improve it.
    * **traits_dataset**: collection of different character traits with corresponding scores.
    * **word_score_dataset**: dataset of common words with their sentiment score.
- **src/**: files with code.
    * **sent_model/**:
        * sentiment_test.py: used to predict a sentence with sentiment transformer.
        * sentiment_train.py: code to fine-tune our sentiment trasnformer.
    * **utils/**:
        * **ROCStoryDataset**: and StoryDataset: classes to represent individual dataset
        * **constants.py**: Changable fine-tunning constants, such as amount of epoch, sliding window token size, ...
        * **Evaluation.py**: function used for evaluation
        * **SentimentAnalysis.py**: code used to analyze sentiment of given story.
        * **StoryEndingGeneration.py**: Story ending transformer code  
    * **run.ipynb**: Jupyter notebook to test our model for generating ending of stories by considering sentiment  
    * **main.py**: Python file we use to generate text from given promp
- **generated.md**: Examples of testing our fine-tuned GPT2 model with Sentiment analysis
- **requirements.txt**: text file containing all necessary packages for generating ending of stories
