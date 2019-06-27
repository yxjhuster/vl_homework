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
        self.feature = googlenet(pretrained=True)
        self.embedding = nn.Linear(num_question, 1500)
        self.fc = nn.Linear(2524, num_answer)

    def forward(self, image, question_encoding):
        # pdb.set_trace()
        features = self.feature(image)
        embeddings = self.embedding(question_encoding)
        concat_features = torch.cat((features, embeddings), dim=1)

        output = self.fc(concat_features)
        return output
