import torch.nn as nn
import torch.nn.functional as F


class DeepSpeech(nn.Module):
    def __init__(self, vocab_size):
        super(DeepSpeech, self).__init__()
        self.vocab_size = vocab_size
        self.dropout1 = nn.Dropout(p=0.4)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=3)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=30, kernel_size=3, stride=3)
        self.conv2_bn = nn.BatchNorm2d(30)
        self.lstm_1 = nn.LSTM(input_size=510, hidden_size=200, bidirectional=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.lstm2 = nn.LSTM(input_size=400, hidden_size=100, bidirectional=True)
        self.fc_bn = nn.BatchNorm1d(11)
        self.fc = nn.Linear(in_features=200, out_features=vocab_size)

    def forward(self, input_):
        x = self.dropout1(input_)
        x = self.conv1_bn(F.relu(self.conv1(x)))
        x = self.conv2_bn(F.relu(self.conv2(x)))
        N, C, F_, T = x.size()
        x = x.view(N, C * F_, T)  # flatten on feature maps axis
        x = x.permute(2, 0, 1)  # Sequence, Batch, Features
        x, _ = self.lstm_1(x)
        x = self.dropout2(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x.permute(1, 0, 2)  # transpose to Batch, Sequence, Features
        x = self.fc_bn(x)
        x = F.relu(self.fc(x))
        x = F.log_softmax(x, dim=2)
        x = x.permute(1, 0, 2)  # transpose again but now to Sequence, Batch, Features - this is mandatory for CTC loss
        return x
