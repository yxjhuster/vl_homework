import torch
import torch.nn as nn
import pdb


class ParallelAttention(nn.Module):
    def __init__(self, embedding_size, ques_seq_size, img_seq_size):
        super().__init__()
        self.embedding_size = embedding_size

        self.ques_seq_size = ques_seq_size
        self.img_seq_size = img_seq_size

        self.k = 512
        # self.output_size_ques = self.ques_seq_size * self.k
        # self.output_size_img = self.img_seq_size * self.k

        self.img_corr_dim = nn.Linear(
            self.embedding_size, self.embedding_size)
        self.ques_embed_dim = nn.Linear(
            self.embedding_size, self.k)
        self.img_embed_dim = nn.Linear(
            self.embedding_size, self.k)
        # self.ques_atten_embedding = nn.Linear(
        #     self.k * self.ques_seq_size, self.ques_seq_size)
        self.ques_atten_embedding = nn.Linear(
            self.k, 1)
        # self.img_atten_embedding = nn.Linear(
        #     self.k * self.img_seq_size, self.img_seq_size)
        self.img_atten_embedding = nn.Linear(
            self.k, 1)

        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(0)
        self.softmax = nn.Softmax()

    def forward(self, ques_feat, img_feat, mask):
        # pdb.set_trace()
        b_s = ques_feat.shape[0]
        img_feat = img_feat.view(b_s, self.embedding_size, -1)
        # Wb * V
        img_corr_dim_value = self.img_corr_dim(img_feat.permute(0, 2, 1))
        img_corr = img_corr_dim_value.permute(0, 2, 1)
        # tanh(Q' * Wb * V)
        weight_matrix = self.tanh(torch.bmm(ques_feat, img_corr))

        # pdb.set_trace()
        # Wq * Q
        ques_embed_dim_value = self.ques_embed_dim(ques_feat)
        ques_embed = ques_embed_dim_value.permute(0, 2, 1)

        # Wv * V
        img_embed_dim_value = self.img_embed_dim(img_feat.permute(0, 2, 1))
        img_embed = img_embed_dim_value.permute(0, 2, 1)

        # Wq * Q * C
        transform_ques = torch.bmm(ques_embed, weight_matrix)

        # Hv
        img_atten_sum = self.drop(self.tanh(transform_ques + img_embed))
        # img_atten_embedding_value = self.img_atten_embedding(
        #     img_atten_sum.reshape([b_s, self.k * self.img_seq_size]))

        # pdb.set_trace()
        img_atten_embedding_value = self.img_atten_embedding(
            img_atten_sum.permute(0, 2, 1))
        img_atten_embedding_value = torch.squeeze(img_atten_embedding_value)

        # softmax
        img_atten = self.softmax(
            img_atten_embedding_value.reshape([b_s, self.img_seq_size]))

        # Wv * V * C'
        transform_img = torch.bmm(img_embed, weight_matrix.permute(0, 2, 1))
        ques_atten_sum = self.drop(self.tanh(transform_img + ques_embed))
        # ques_atten_embedding_value = self.ques_atten_embedding(
        #     ques_atten_sum.reshape([b_s, self.k * self.ques_seq_size]))
        ques_atten_embedding_value = self.ques_atten_embedding(
            ques_atten_sum.permute(0, 2, 1))
        ques_atten_embedding_value = torch.squeeze(ques_atten_embedding_value)
        # pdb.set_trace()
        ques_atten = self.softmax(
            (ques_atten_embedding_value * mask).reshape([b_s, self.ques_seq_size]))

        # pdb.set_trace()

        ques_atten_dim = ques_atten.reshape([b_s, 1, -1])
        img_atten_dim = img_atten.reshape([b_s, 1, -1])

        ques_atten_feat = torch.bmm(ques_atten_dim, ques_feat)
        ques_atten_feat = torch.squeeze(ques_atten_feat)
        img_atten_feat = torch.bmm(img_atten_dim, img_feat.permute(0, 2, 1))
        img_atten_feat = torch.squeeze(img_atten_feat)

        return ques_atten_feat, img_atten_feat
