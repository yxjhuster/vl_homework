import torch.nn as nn
import torch
# from torch.autograd import Variable
import pdb
import sys
sys.path.append('../')
from external.googlenet.googlenet import googlenet


class SimpleBaselineNet(nn.Module):
    """
    Predicts an answer to a question about an image using the Simple Baseline for Visual Question Answering (Zhou et al, 2017) paper.
    """

    def __init__(self, num_question, num_answer, pretrained=True):
        super().__init__()
        # output should be N * 1024
        self.feature = googlenet(pretrained=True)
        # output should be N * 1500
        # self.embedding = nn.Embedding(num_question, 1500)
        self.embedding = nn.Linear(num_question, 2000)
        # self.fc = nn.Linear(1024 + 2001, num_answer)
        self.fc = nn.Linear(1024 + 2000, num_answer)

    def forward(self, image, question_encoding):
        features = self.feature(image)
        # embeddings = self.embedding(question_encoding)
        # # pdb.set_trace()
        # embeddings = torch.sum(embeddings, dim=1)
        # embeddings = torch.clamp(embeddings, 0, 1)
        embeddings = self.embedding(question_encoding)
        # pdb.set_trace()
        # print(features.size())
        # print(question_encoding.size())
        # concat_features = torch.cat((features, question_encoding), dim=1)
        concat_features = torch.cat((features, embeddings), dim=1)

        output = self.fc(concat_features)
        return output
