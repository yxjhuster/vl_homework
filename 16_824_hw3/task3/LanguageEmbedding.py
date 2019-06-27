import torch
import torch.nn as nn
from torch.nn.utils import rnn
import pdb
import numpy as np


class LanguageEmbedding(nn.Module):
    def __init__(self, num_question, conv_size, embedding_size, max_len):
        super().__init__()

        # self.embedding = nn.Linear(num_question, embedding_size)
        self.embedding = nn.Linear(num_question, embedding_size)

        self.unigram = nn.Conv1d(embedding_size, conv_size, 1, 1, 0)
        self.bigram = nn.Conv1d(embedding_size, conv_size, 2, 1, 1)
        self.trigram = nn.Conv1d(embedding_size, conv_size, 3, 1, 1)

        self.tanh = nn.Tanh()

        self.lstm = nn.LSTM(input_size=conv_size,
                            hidden_size=conv_size, num_layers=2)

        self.max_seq = torch.tensor(np.zeros((max_len, embedding_size)))

    def forward(self, word_rep):
        embedding_list = [self.embedding(word) for word in word_rep]
        # pdb.set_trace()

        embedding_list.append(self.max_seq)

        embeddings = rnn.pad_sequence(embedding_list, batch_first=True)

        embeddings = embeddings[:-1]

        embeddings = embeddings.permute((0, 2, 1))

        seq_len = embeddings.shape[2]

        unigram_out = self.unigram(embeddings)
        bigram_out = self.bigram(embeddings)
        trigram_out = self.trigram(embeddings)

        bigram_out = bigram_out.narrow(2, 1, seq_len)

        feat = torch.stack((unigram_out, bigram_out, trigram_out), dim=3)

        max_feat, _ = feat.max(dim=3)

        phrase_embeddings = self.tanh(max_feat)

        phrase_embeddings = phrase_embeddings.permute((0, 2, 1))

        question_embeddings, _ = self.lstm(phrase_embeddings.permute(1, 0, 2))

        embeddings = embeddings.permute((0, 2, 1))

        return embeddings, phrase_embeddings, question_embeddings.permute(1, 0, 2)
