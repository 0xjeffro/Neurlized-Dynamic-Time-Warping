from torch import nn


class LSTM_Classification(nn.Module):

    def __init__(self, input_dim, hidden_dim, target_size, num_layers=1, bidirectional=False, dropout=0.0):
        super(LSTM_Classification, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        fc_dim = hidden_dim
        if bidirectional is True:
            fc_dim = fc_dim * 2
        self.fc = nn.Linear(fc_dim, target_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_):
        lstm_out, (h, c) = self.lstm(input_.cuda())
        logits = self.fc(lstm_out[:, -1])
        scores = self.softmax(logits)
        return scores
