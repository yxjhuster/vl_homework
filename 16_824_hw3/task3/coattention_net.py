import torch.nn as nn
import torch

from LanguageEmbedding import LanguageEmbedding
from ParallelAttention import ParallelAttention

import pdb


class CoattentionNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
    for Visual Question Answering (Lu et al, 2017) paper.
    """

    def __init__(self, num_question, num_answer, max_len):
        super().__init__()
        self.num_question = num_question
        self.num_answer = num_answer
        self.embedding_model = LanguageEmbedding(
            self.num_question, 512, 512, max_len)

        self.word_attention = ParallelAttention(512, max_len, 196)
        self.phrase_attention = ParallelAttention(512, max_len, 196)
        self.question_attention = ParallelAttention(512, max_len, 196)

        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512 + 512, 512)
        self.fc3 = nn.Linear(512 + 512, 2001)
        self.fc4 = nn.Linear(2001, self.num_answer)

        self.tanh = nn.Tanh()

    def forward(self, image, question_encoding, mask):
        # TODO
        # raise NotImplementedError()
        embeddings, phrase_embeddings, question_embeddings = self.embedding_model(
            question_encoding)

        # pdb.set_trace()

        q_w, v_w = self.word_attention(embeddings, image, mask)
        phrase_input = q_w + v_w
        h_w = self.fc1(phrase_input)
        h_w = self.tanh(h_w)

        # pdb.set_trace()

        q_p, v_p = self.phrase_attention(phrase_embeddings, image, mask)
        ques_input = torch.cat((h_w, q_p + v_p), dim=1)
        h_p = self.fc2(ques_input)
        h_p = self.tanh(h_p)

        q_s, v_s = self.question_attention(question_embeddings, image, mask)
        cls_input = torch.cat((h_p, q_s + v_s), dim=1)
        cls_input = self.fc3(cls_input)
        cls_input = self.tanh(cls_input)
        cls_output = self.fc4(cls_input)

        return cls_output


# class CoattentionNet(nn.Module):
#     """
#     Predicts an answer to a question about an image using the Hierarchical Question-Image Co-Attention
#     for Visual Question Answering (Lu et al, 2017) paper.
#     """

#     def __init__(self, num_question, num_answer, max_len):
#         super().__init__()
#         self.num_question = num_question
#         self.num_answer = num_answer
#         self.embedding_model = LanguageEmbedding(
#             self.num_question, 64, 64, max_len)

#         self.word_attention = ParallelAttention(64, max_len, 32)
#         self.phrase_attention = ParallelAttention(64, max_len, 32)
#         self.question_attention = ParallelAttention(64, max_len, 32)

#         self.fc1 = nn.Linear(64, 64)
#         self.fc2 = nn.Linear(64 + 64, 64)
#         self.fc3 = nn.Linear(64 + 64, 64)
#         self.fc4 = nn.Linear(64, self.num_answer)

#     def forward(self, image, question_encoding):
#         # TODO
#         # raise NotImplementedError()
#         embeddings, phrase_embeddings, question_embeddings = self.embedding_model(
#             question_encoding)

#         q_w, v_w = self.word_attention(embeddings, image)
#         phrase_input = q_w + v_w
#         h_w = self.fc1(phrase_input)

#         # pdb.set_trace()

#         q_p, v_p = self.phrase_attention(phrase_embeddings, image)
#         ques_input = torch.cat((h_w, q_p + v_p), dim=1)
#         h_p = self.fc2(ques_input)

#         q_s, v_s = self.question_attention(question_embeddings, image)
#         cls_input = torch.cat((h_p, q_s + v_s), dim=1)
#         cls_input = self.fc3(cls_input)
#         cls_output = self.fc4(cls_input)

#         return cls_output
