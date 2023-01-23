from torch import nn


class GRU_Classification(nn.Module):
    def __init__(self, input_dim, hidden_dim, target_size, num_layers=1, bidirectional=False, dropout=0.0):
        super(GRU_Classification, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True,
                          num_layers=num_layers,
                          bidirectional=bidirectional,
                          dropout=dropout)
        fc_dim = hidden_dim
        if bidirectional is True:
            fc_dim = fc_dim * 2
        self.fc = nn.Linear(fc_dim, target_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_):
        gru_out, h = self.gru(input_.cuda())
        logits = self.fc(gru_out[:, -1])

        scores = self.softmax(logits)
        return scores