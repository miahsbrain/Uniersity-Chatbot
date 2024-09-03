import torch
from torch import nn
from torch.nn import functional as F
import nltk
from nltk.stem import PorterStemmer
import json
import random
import string

with open('university_support.json', 'r') as f:
    data = json.load(f)

allwords = []
tags = []
xy = []

stemmer = PorterStemmer()

def tokenize(text: str):
    return nltk.tokenize.word_tokenize(text)

def stem(input: list):
    return stemmer.stem(input)

def bag_of_words(allwords: list, input: list) -> torch.tensor:
    word = torch.zeros(size=[len(allwords)])
    for idx, text in enumerate(allwords):
        if text in input:
            word[idx] = 1
    return word

for intents in data['intents']:
    if intents['tag'] not in tags:
        tag = intents['tag']
        tags.append(tag)
    
    for patterns in intents['patterns']:
        word = tokenize(patterns.lower())
        word = [stem(x) for x in word if x not in list(string.punctuation)]
        allwords.extend(word)
        xy.append(([i for i in word], tag))

allwords = sorted(set(allwords))

# Hyper parameters
input_features = len(allwords)
output_features = len(tags)
hidden_units = 256

class ChatbotV1(nn.Module):
    def __init__(self, input_features, output_features, hidden_units):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=input_features, out_features=hidden_units)
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.layer_3 = nn.Linear(in_features=hidden_units, out_features=hidden_units)
        self.classifier = nn.Linear(in_features=hidden_units, out_features=output_features)


    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.classifier(F.relu(self.layer_3(F.relu(self.layer_2(F.relu(self.layer_1(x)))))))

model_v1 = ChatbotV1(input_features=input_features, output_features=output_features, hidden_units=hidden_units)
model_v1.load_state_dict(torch.load('ChatbotV1'))

def chat(text):
    text = tokenize(text)
    text = [stem(w) for w in text]
    text = bag_of_words(allwords, text)

    model_v1.eval()
    with torch.inference_mode():
        logit = model_v1(text)

    pred = torch.argmax(torch.softmax(logit, dim=-1))
    c_tag = tags[pred]
    # print(f'Index: {pred} | Class: {c_tag}')
    res = [random.choice(x["responses"]) for x in data["intents"] if x["tag"] == c_tag][0]
    # print(f'Response: {res}')
    return res
    

chat('Who created you')