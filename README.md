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
# TODO :
Result : Device set to use cuda:0
[{'label': 'POSITIVE', 'score': 0.9998712539672852}]
```
Analysis : 
The sentiment analysis classifier accurately detects the positive tone in the given sentence. It shows a high confidence score, indicating that the model is reliable for straightforward emotional expressions, such as enthusiasm or joy, in English-language input.
### 2. Example 2 - Text Generation

```python
# TODO :
generator = pipeline("text-generation", model="gpt2")
generator("In this project, we are exploring how to use transformers for")
```

### 3. Example 3 - Named Entity Recognition

```python
# TODO :
ner = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
ner("My name is Sarah Jessica Parker but you can call me Jessica")
```

## Project Overview

This project demonstrates the use of Hugging Face Transformers for various Natural Language Processing tasks. The examples showcase different capabilities of transformer models including sentiment analysis, text generation, and named entity recognition.

## Requirements

- Python 3.7+
- PyTorch 1.8+
- Transformers 4.5+
- Datasets 1.5+

## Installation

```bash
pip install transformers torch datasets
```

## Usage

Each example in the "My todo" section demonstrates a different NLP task using Hugging Face's pipeline API. To run these examples, simply copy the code into a Python script or Jupyter notebook.

## Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Transformers GitHub Repository](https://github.com/huggingface/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
