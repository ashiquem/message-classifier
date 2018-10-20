# Message Classifier

A simple text classification project, which learns from chat logs of you and your friend to predict how much a message sounds like either of you.

## Dependencies

- Numpy
- Sci-kit learn
- Scipy

## Usage

### Step 1

Clone the repo. Download the chat log from Whatsapp without media and place the text file in a folder. e.g. a folder named 'data'

### Step 2

Run `classify.py` passing the name of the folder containing your chat log text file. For example:

```python classify.py data```

The app will train the model and start an interactive session with you asking for your input to classify. Type in some messages and have fun!

Currently supports Whatsapp chat logs only. You can use the `message_cleaner.py` on its own to process Whatsapp chat logs. To use it, pass in the name of the folder and the csv file to be saved. For example:

```python message_cleaner.py data my_messages.csv```

## Improvements

The current model is a Naive Bayes - SVM, based on the paper [Baselines and Bigrams: Simple, Good Sentiment and Topic Classiﬁcation.](https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf) It's a really simple method, very useful as a baseline. I came across this in a [lecture](https://youtu.be/37sFIak42Sc) video by Jeremy Howard, definitely worth taking a look if you are new to NLP. I will be using this project to play around with more advanced NLP techniques. I plan to try out the following sometime soon:

- LSTM
- LSTM + NB-SVM
- Bi-directional LSTM

I will update this section as I discover interesting approaches.