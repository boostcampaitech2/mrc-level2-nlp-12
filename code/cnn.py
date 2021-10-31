import torch
import torch.nn as nn

# http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-embeddings/
class Conv1dHeadConcat(nn.Module):
    '''A class as a head of PLM for convolutional process

    Args:
        hidden_size (int): a value from PLM hidden size
    '''
    def __init__(self, hidden_size):
        super().__init__()

        # output size => same / stride = 1 (default)
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=5, padding=2)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=256 * 3, out_features=2) # 2 => start and end logits

    def forward(self, x):
        x = x.transpose(1, 2).contiguous() # (B x hidden size x 512)

        # squeeze => (B x 512 x hidden size x 1) to (B x 512 x hidden size)
        conv1 = self.relu(self.conv1(x).transpose(1, 2).contiguous().squeeze(-1)) # conv => gelu, mish possible
        conv2 = self.relu(self.conv2(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3 = self.relu(self.conv3(x).transpose(1, 2).contiguous().squeeze(-1))
        dropout = self.dropout(torch.cat((conv1, conv2, conv3), -1))
        output = self.fc(dropout)
        return output

class Conv1dHeadSum(nn.Module):
    '''A class as a head of PLM for convolutional process

    Args:
        hidden_size (int): a value from PLM hidden size
    '''
    def __init__(self, hidden_size):
        super().__init__()

        # output size => same / stride = 1 (default)
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=2, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_size, out_channels=2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_size, out_channels=2, kernel_size=5, padding=2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2).contiguous() # (B x hidden size x 512) => conv layer rule

        # squeeze => (B x 512 x hidden size x 1) to (B x 512 x hidden size)
        conv1 = self.relu(self.conv1(x).transpose(1, 2).contiguous().squeeze(-1)) # conv => relu activation function
        conv2 = self.relu(self.conv2(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3 = self.relu(self.conv3(x).transpose(1, 2).contiguous().squeeze(-1))
        return conv1 + conv2 + conv3