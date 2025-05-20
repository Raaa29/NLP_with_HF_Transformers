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

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers 4.5+
- Datasets 1.5+

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers GitHub Repository](https://github.com/huggingface/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
