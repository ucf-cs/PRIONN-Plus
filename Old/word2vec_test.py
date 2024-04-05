# TODO: This never worked well and was usurped by gensim word2ec. Consider deleting.
# Import necessary libraries
import codecs
import numpy as np
import os
import pickle
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics.pairwise import cosine_similarity


def load_corpus(directory):
    corpus = ""
    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        # Check if file is a text file
        if filename.endswith(".txt"):
            # Open the file in read mode and append its content to the corpus
            with codecs.open(os.path.join(directory, filename), 'r', encoding='utf8') as file:
                corpus += file.read() + " "
    return corpus

# Preprocess the text data, remove stop words and punctuation, and create a vocabulary of unique words


def preprocessing(corpus, window_size):
    # stop_words = set(stopwords.words('english'))
    raw_text = corpus.split()
    # Convert to lowercase, remove stop words and punctuation
    text = [word.lower() for word in raw_text if word not in string.punctuation]

    # Create a vocabulary of unique words
    vocab = set(text)
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    print("Found " + str(vocab_size) + " words in the corpus.")

    # Create training pairs according to window size
    data = []
    for i in range(window_size, len(text) - window_size):
        context = [text[i - j - 1]
                   for j in range(window_size)] + [text[i + j + 1] for j in range(window_size)]
        target = text[i]
        data.append((target, context))

    return data, vocab_size, word_to_ix, text

# Define the skip-gram model


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = self.linear(embeds)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

# Define the training process


def train_model(data, vocab_size, word_to_ix, EMBEDDING_DIM, EPOCHS):
    if os.path.isfile("model/test.pk"):
        return pickle.load(open("model/test.pk", 'rb'))

    losses = []
    loss_function = nn.NLLLoss()
    model = SkipGram(vocab_size, EMBEDDING_DIM)  # .to("mps")
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        print("Running epoch " + str(epoch))
        total_loss = 0
        for target, context in data:
            target_idx = torch.tensor(
                [word_to_ix[target]], dtype=torch.long)  # , device="mps")
            for context_word in context:
                context_idx = torch.tensor(
                    [word_to_ix[context_word]], dtype=torch.long)  # , device="mps")
                model.zero_grad()
                log_probs = model(target_idx)
                loss = loss_function(log_probs, context_idx)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        losses.append(total_loss)

    if not os.path.exists("model"):
        os.makedirs("model")
    if not os.path.exists("model/test.pk"):
        pickle.dump(model, open("model/test.pk", 'wb'))

    return model

# Define the prediction process


def get_vector(word, word_to_ix, model):
    word_tensor = torch.tensor([word_to_ix[word]], dtype=torch.long)
    return model.embeddings(word_tensor).data.numpy()

def find_similar_word(target_word, embeddings_dict):
    # Get the embedding for the target word
    target_embedding = embeddings_dict[target_word]
    return find_similar(target_embedding, embeddings_dict)

def find_similar(target_embedding, embeddings_dict):
    # Calculate the cosine similarity between the target embedding and all other embeddings
    similarities = {word: cosine_similarity(target_embedding.reshape(1, -1), embedding.reshape(1, -1))[0][0]
                    for word, embedding in embeddings_dict.items()}

    # Sort the words by their similarity to the target word
    sorted_similarities = sorted(
        similarities.items(), key=lambda x: x[1], reverse=False)

    # Return the top 5 most similar words
    return [word for word, similarity in sorted_similarities[:5]]


corpus = load_corpus("data")
window_size = 2
data, vocab_size, word_to_ix, text = preprocessing(corpus, window_size)
print("Corpus preprocessed!")
EMBEDDING_DIM = 4
EPOCHS = 10
model = train_model(data, vocab_size, word_to_ix, EMBEDDING_DIM, EPOCHS)
print("Model trained!")

embeddings = {}
for word in text:
    embeddings[word] = get_vector(word, word_to_ix, model)
print("Word vectors retrieved!")

doc = get_vector("doctor", word_to_ix, model)
man = get_vector("man", word_to_ix, model)
woman = get_vector("woman", word_to_ix, model)
woman_doc = doc - man + woman
print("doctor - man + woman = " + str(find_similar(woman_doc, embeddings)))

for word in text:
    # print(embeddings[word])
    print(word, find_similar_word(word, embeddings))
