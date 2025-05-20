# NLP_with_HF_Transformers
# Natural Language Processing with Hugging Face Transformers

Generative AI Guided Project on Cognitive Class by IBM

[![Python](https://img.shields.io/badge/PYTHON-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PYTORCH-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)

## Name : Farraheira 

## My todo :

### 1. Example 1 - Sentiment Analysis

```python
# TODO :
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("i Like him because he's good person i ever meet")
```
```
# Result :
Device set to use cuda:0
[{'label': 'POSITIVE', 'score': 0.9998712539672852}]
```
Analysis : 
The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.

### 2. Example 2 -  Topic Classification

```python
# TODO :
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
classifier(
    "Exploratory Data Analysis is the first course in Machine Learning Program that introduces learners to the broad range of Machine Learning concepts, applications, challenges, and solutions, while utilizing interesting real-life datasets",
    candidate_labels=["art", "natural science", "data analysis"],
)
```
```
# Result :
{'sequence': 'Exploratory Data Analysis is the first course in Machine Learning Program that introduces learners to the broad range of Machine Learning concepts, applications, challenges, and solutions, while utilizing interesting real-life datasets',
 'labels': ['data analysis', 'art', 'natural science'],
 'scores': [0.9957792162895203, 0.0026982570998370647, 0.0015224907547235489]}
```
Analysis : 
In summary, the model successfully identified "data analysis" as the most relevant topic for the provided text about an Exploratory Data Analysis course, with a high degree of certainty.

### 3. Example 3 - Text Generator

```python
# TODO :
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "This course will teach you",
    max_length=30,
    num_return_sequences=2,
)
```
```
# Result :
[{'generated_text': "This course will teach you how to learn how to handle a lot of these features using WordPress on Rails. You should find one that's right in your"},
 {'generated_text': 'This course will teach you how to do most of the things that you wouldn\u202ct like to do. You can also watch videos of me doing'}]
```
Analysis :
In summary, the model generated two different possible continuations for the sentence "This course will teach you", providing examples of how a text generation model can creatively complete a given prompt.

### 3. Example 3 - Text Generator

```python
# TODO :
generator = pipeline("text-generation", model="distilgpt2")
generator(
    "This course will teach you",
    max_length=30,
    num_return_sequences=2,
)
```
```
# Result :
[{'generated_text': "This course will teach you how to learn how to handle a lot of these features using WordPress on Rails. You should find one that's right in your"},
 {'generated_text': 'This course will teach you how to do most of the things that you wouldn\u202ct like to do. You can also watch videos of me doing'}]
```
Analysis :
In summary, the model generated two different possible continuations for the sentence "This course will teach you", providing examples of how a text generation model can creatively complete a given prompt.

### 4. Example 4 - Name Entity Recognition (NER)

```python
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", grouped_entities=True)
ner("My name is Roberta and I work with IBM Skills Network in Toronto")
)
```
```
# Result :
Device set to use cuda:0
[{'entity_group': 'PER',
  'score': np.float32(0.9993105),
  'word': 'Roberta',
  'start': 11,
  'end': 18},
 {'entity_group': 'ORG',
  'score': np.float32(0.9976597),
  'word': 'IBM Skills Network',
  'start': 35,
  'end': 53},
 {'entity_group': 'LOC',
  'score': np.float32(0.99702173),
  'word': 'Toronto',
  'start': 57,
  'end': 64}]
```
Analysis :
 the NER model successfully extracted and classified the person's name, the organization's name, and the location from the input sentence.
 
### 5. Example 5 - Question Answering

```python
# TODO :
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What is the capital of France?"
context = "Paris is the capital and most populous city of France, with an estimated population of 2,140,526 residents as of 2020."
qa_model(question = question, context = context)
```
```
# Result :
{'score': 0.9940207600593567, 'start': 0, 'end': 5, 'answer': 'Paris'}
```
Analysis :
the question answering model successfully identified "Paris" as the answer to the question based on the provided context, with a high degree of confidence.

### 6. Example 6 - Text Summarization

```python
# TODO :
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summarizer(
    """
Exploratory Data Analysis is the first course in Machine Learning Program that introduces learners to the broad range of Machine Learning concepts, applications, challenges, and solutions, while utilizing interesting real-life datasets. So, what is EDA and why is it important to perform it before we dive into any analysis?
EDA is a visual and statistical process that allows us to take a glimpse into the data before the analysis. It lets us test the assumptions that we might have about the data, proving or disproving our prior believes and biases. It lays foundation for the analysis, so our results go along with our expectations. In a way, it’s a quality check for our predictions.
As any data scientist would agree, the most challenging part in any data analysis is to obtain a good quality data to work with. Nothing is served to us on a silver plate, data comes in different shapes and formats. It can be structured and unstructured, it may contain errors or be biased, it may have missing fields, it can have different formats than what an untrained eye would perceive. For example, when we import some data, very often it would contain a time stamp. To a human it is understandable format that can interpreted. But to a machine, it is not interpretable, so it needs to be told what that means, the data needs to be transformed into simple numbers first. There are also different date-time conventions depending on a country (i.e., Canadian versus USA), metric versus imperial systems, and many other data features that need to be recognized before we start doing the analysis. Therefore, the first step before performing any analysis – is get really aquatinted with your data!
This course will teach you to ‘see’ and to ‘feel’ the data as well as to transform it into analysis-ready format. It is introductory level course, so no prior knowledge is required, and it is a good starting point if you are interested in getting into the world of Machine Learning. The only thing that is needed is some computer with internet, your curiosity and eagerness to learn and to apply acquired knowledge.  If you live in Canada, you might be interested about gasoline prices in different cities or if you are an insurance actuary you need to analyze the financial risks that you will take based on your clients information. Whatever is the case, you will be able to do your own analysis, and confirm or disprove some of the existing information.
The course contains videos and reading materials, as well as well as a lot of interactive practice labs that learners can explore and apply the skills learned. It will allow you to use Python language in Jupyter Notebook, a cloud-based skills network environment that is pre-set for you with all available to be downloaded packages and libraries. It will introduce you to the most common visualization libraries such as Pandas, Seaborn, and Matplotlib to demonstrate various EDA techniques with some real-life datasets.

"""
)
)
```
```
# Result :
[{'summary_text': ' Exploratory Data Analysis is the first course in Machine Learning Program that introduces learners to the broad range of Machine Learning concepts, applications, challenges, and solutions . EDA is a visual and statistical process that allows us to take a glimpse into the data before the analysis . It lays foundation for the analysis so our results go along with our expectations .'}]
```
Analysis :
the summarization model successfully condensed the provided text about EDA into a shorter summary, retaining the key points.

### 7. Example 7 - Translation

```python
# TODO :
translator_id = pipeline("translation", model="Helsinki-NLP/opus-mt-id-fr")
translator_id("aku pengen tidur")
```
```
# Result :
Device set to use cuda:0
[{'translation_text': 'Je veux dormir.'}]
```
Analysis :
The translated text "Je veux dormir" is the French translation of the Indonesian sentence "aku pengen tidur".
In summary, the translation model successfully translated the Indonesian sentence into French.

## Project Overview

This project demonstrates the use of Hugging Face Transformers for various Natural Language Processing tasks. The examples showcase different capabilities of transformer models including sentiment analysis, text generation, and named entity recognition.

## Exploring the Universe of NLP

![Astronaut exploring space with AI](data:image/jpeg;base64,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)

*Natural Language Processing is like exploring a new universe of possibilities - just as astronauts venture into space, we're venturing into the frontiers of human-machine communication.*

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers 4.5+
- Datasets 1.5+

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers GitHub Repository](https://github.com/huggingface/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
