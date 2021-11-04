import torch
import torch.nn as nn

# http://www.joshuakim.io/understanding-how-convolutional-neural-network-cnn-perform-text-classification-with-word-embeddings/
# https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM

# LSTM + Conv1d (concat)
class Conv1dHeadConcat(nn.Module):
    """A class as a head of PLM for convolutional process
    Args:
        hidden_size (int): a value from PLM hidden size
    """

    def __init__(self, hidden_size):
        super().__init__()

        # output size => same / stride = 1 (default)
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=256, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size, out_channels=256, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=hidden_size, out_channels=256, kernel_size=5, padding=2
        )

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(
            in_features=256 * 3, out_features=2
        )  # 2 => start and end logits

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()  # (B x hidden size x 512)

        # squeeze => (B x 512 x hidden size x 1) to (B x 512 x hidden size)
        conv1 = self.relu(
            self.conv1(x).transpose(1, 2).contiguous().squeeze(-1)
        )  # conv => gelu, mish possible
        conv2 = self.relu(self.conv2(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3 = self.relu(self.conv3(x).transpose(1, 2).contiguous().squeeze(-1))
        dropout = self.dropout(torch.cat((conv1, conv2, conv3), -1))
        output = self.fc(dropout)
        return output


# LSTM + Conv1d (summation)
class Conv1dHeadSum(nn.Module):
    """A class as a head of PLM for convolutional process
    Args:
        hidden_size (int): a value from PLM hidden size
    """

    def __init__(self, hidden_size):
        super().__init__()

        # output size => same / stride = 1 (default)
        self.conv1 = nn.Conv1d(in_channels=hidden_size, out_channels=2, kernel_size=1)
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size, out_channels=2, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=hidden_size, out_channels=2, kernel_size=5, padding=2
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()  # (B x hidden size x 512) => conv layer rule

        # squeeze => (B x 512 x hidden size x 1) to (B x 512 x hidden size)
        conv1 = self.relu(
            self.conv1(x).transpose(1, 2).contiguous().squeeze(-1)
        )  # conv => relu activation function
        conv2 = self.relu(self.conv2(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3 = self.relu(self.conv3(x).transpose(1, 2).contiguous().squeeze(-1))
        return conv1 + conv2 + conv3


# LSTM + Conv1d with GELU
class LSTMConv1dHeadConcat(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gelu = nn.GELU()
        self.biLSTM = nn.LSTM(
            input_size=hidden_size,  # output from PLM
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.4,
            batch_first=True,
            bidirectional=True,  # output dim x 2
        )

        # conv1d => hidden_size * 2 from biLSTM
        self.conv1 = nn.Conv1d(
            in_channels=hidden_size * 2, out_channels=256, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=hidden_size * 2, out_channels=256, kernel_size=3, padding=1
        )
        self.conv3 = nn.Conv1d(
            in_channels=hidden_size * 2, out_channels=256, kernel_size=5, padding=2
        )
        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(in_features=512 * 3, out_features=2)

    def forward(self, x):
        # (output, hidden state, cell state)
        lstm_output, (_, _) = self.biLSTM(x)  # output => (B, L, 2 x D)
        x = lstm_output.transpose(1, 2).contiguous()
        conv1 = self.gelu(self.conv1(x).transpose(1, 2).contiguous().squeeze(-1))
        conv2 = self.gelu(self.conv2(x).transpose(1, 2).contiguous().squeeze(-1))
        conv3 = self.gelu(self.conv3(x).transpose(1, 2).contiguous().squeeze(-1))
        dropout = self.dropout(torch.cat((conv1, conv2, conv3), -1))
        output = self.fc(dropout)
        return output
