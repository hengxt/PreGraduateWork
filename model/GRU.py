import torch.nn as nn

class GRUCore(nn.Module):
    def __init__(self, input_size, hidden_size)
        super().__init__()

        self.Uz = nn.Linear(input_size, hidden_size)
        self.Ur = nn.Linear(input_size, hidden_size)
        self.Uo = nn.Linear(input_size, hidden_size)

        self.Vz = nn.Linear(hidden_size, hidden_size)
        self.Vr = nn.Linear(hidden_size, hidden_size)
        self.Vo = nn.Linear(hidden_size, hidden_size)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs)
        batch_len = inputs.shape[0]

        h_t_pre = Variable(torch.zeros(batch_len, hidden_size))
        zt = self.sigmoid(torch.add(self.Uz(inputs), self.Vz(h_t_pre)))
        rt = self.sigmoid(torch.add(self.Ur(inputs), self.Vr(h_t_pre)))
        ht1 = self.tanh(torch.add(self.Uo(inputs), torch.mul(rt, self.Vo(h_t_pre))))    
        tmp = torch.ones(batch_len, hidden_size) - zt
        ht = torch.add(torch.mul(h_t_pre, tmp), torch.mul(zt, ht1))

        return ht