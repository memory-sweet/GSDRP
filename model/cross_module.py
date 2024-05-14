import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, d_model):
        super(CrossAttention, self).__init__()

        self.query_linear = nn.Linear(feature_dim1, d_model)
        self.key_linear = nn.Linear(feature_dim2, d_model)
        self.value_linear = nn.Linear(feature_dim2, d_model)

        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to('cuda:0')

    def forward(self, Data1, Data2):

        query = self.query_linear(Data1).to('cuda')
        key = self.key_linear(Data2).to('cuda:0')
        value = self.value_linear(Data2).to('cuda:0')

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        attention = F.softmax(attention_scores, dim=-1)

        weighted_value = torch.matmul(attention, value)

        return weighted_value

