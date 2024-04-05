# TODO: Unused. Usurped by gensim word2vec library. Consider deleting this.

# word2vec.py
import torch
import torch.nn as nn
import torch.optim as optim


class Word2Vec(nn.Module):
    """ Define the Word2Vec model. """

    def __init__(self, vocab_size, embedding_size):
        super(Word2Vec, self).__init__()
        # nn.Embedding is used to create the word embeddings with vocab_size and embedding_size.
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)

    # Forward pass.
    def forward(self, inputs):
        """ Get the embeddings of the inputs """
        embeds = self.embeddings(inputs)
        logits = self.linear(embeds)
        return logits

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.linear = nn.Linear(embedding_size, vocab_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        logits = self.linear(embeds)
        out = self.softmax(logits)
        return out


# Example usage
# Assuming you have a list of strings as input
input_strings = ["example input string 1", "example input string 2 here now", "Another example string"]

# Preprocess the input strings and obtain the vocabulary and word_to_ix mapping
vocab = set()
for string in input_strings:
    words = string.split()
    vocab.update(words)
word_to_ix = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
vocab_size = len(vocab)
# TODO: Tune.
embedding_size = 4

# Convert the input strings to tensors of word indices
input_tensors = []
for string in input_strings:
    words = string.split()
    indices = [word_to_ix[word] for word in words]
    input_tensors.append(torch.LongTensor(indices))

# # Initialize the Word2Vec model
# model = Word2Vec(vocab_size, embedding_size)

# Convert the input tensors to padded sequences
padded_input = nn.utils.rnn.pad_sequence(input_tensors)
print(padded_input)

# Initialize the SkipGram model
model = SkipGram(vocab_size, embedding_size)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training
for epoch in range(100):
    total_loss = 0
    for context, target in input_tensors:
        context_var = torch.tensor([word_to_ix[context]], dtype=torch.long)
        model.zero_grad()
        log_probs = model(context_var)
        loss = loss_function(log_probs, torch.tensor([word_to_ix[target]], dtype=torch.long))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Loss at epo {epoch}: {total_loss/len(input_tensors)}")

# Get the word embeddings for the input strings
model.eval()
with torch.no_grad():
    embeddings = model.embeddings(padded_input)

# Print the word embeddings
print(embeddings)


# # Get the word embeddings for the input strings
# model.eval()
# with torch.no_grad():
#     # NOTE: embeddings[word_idx][str_idx][vector_idx]
#     embeddings = model.embeddings(padded_input)
#     results = torch.unbind(embeddings, dim=1)

# # Print the final results.
# print(results)

