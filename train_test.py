import random
import itertools
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from graph_gen import SinGraph


LEARNING_RATE = 0.005
BATCH_SIZE = 100
NUM_EPOCHS = 250
SEQUENCE_LENGTH = 75
HIDDEN_NEURONS=8
NUM_LAYERS = 1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TYPE = nn.RNN # nn.GRU or nn.GRU


class MyRNN(nn.Module):
    def __init__(self, hidden_size, num_layers=NUM_LAYERS, dropout_rate=0.2):
        super(MyRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #
        self.rnn = TYPE(input_size=1, hidden_size=hidden_size, num_layers=self.num_layers, dropout=dropout_rate)
        self.out = nn.Linear(hidden_size, 1)

    def init_hidden_state(self, batch_size):
        self.hidden_state = torch.zeros([self.num_layers, batch_size, self.hidden_size]).to(DEVICE)

    # input of shape (seq_len, batch, input_size):
    # hidden_state of shape (num_layers * num_directions, batch, hidden_size):
    def forward(self, x):
        result, self.hidden_state = self.rnn(x, self.hidden_state)
        result = self.out(result[:, :, :])
        return result


class MyDataset(Dataset):

    # split is an array of [(x0, y0), ... , (xn, yn)]
    def __init__(self, split, seq_length):
        self.split = split
        self.seq_length = seq_length

    def __len__(self):
        return len(self.split) - self.seq_length - 1

    def __getitem__(self, idx):
            # [(x0 y0), (x1 y1), .., (xn yn)]
            input = self.split[idx:idx + self.seq_length]

            # xn+1 yn+1
            output = self.split[idx+1:idx + self.seq_length+1]

            return np.array(input, dtype='float32'), np.array(output, dtype='float32')


def train_model(model, dataloader, loss_function, optimizer, epochs):
    model.train()
    loss_all = []

    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            model.init_hidden_state(BATCH_SIZE)
            optimizer.zero_grad()

            x_batch = x_batch[:,:,np.newaxis].permute([1, 0, 2])
            y_batch = y_batch[:,:,np.newaxis].permute([1, 0, 2])

            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            output = model(x_batch)

            loss = loss_function(output, y_batch)

            loss.backward()
            optimizer.step()

        loss_all.append(loss.cpu().data.numpy())

        print('Training loss for epoch {} : '.format(epoch), loss.cpu().data.numpy())


def test_model(model, dataloader, init_sequence_length):
    model.eval()

    model.init_hidden_state(1)

    batch_input, batch_y = dataloader.dataset[0]

    initial_input = torch.Tensor(batch_input[:, np.newaxis, np.newaxis]).to(DEVICE)

    final_outputs = []

    output = model(initial_input)
    output = output[-1, :, :]
    final_outputs.append(output.cpu().data.squeeze_())
    output = output[np.newaxis, :, :]

    for _ in range(len(dataloader.dataset.split)-init_sequence_length):
        output = model(output)
        final_outputs.append(output.cpu().data.squeeze_())

    def myplot(points, label_name):
        plt.plot(points, linestyle='--', marker='.', label=label_name)

    myplot(dataloader.dataset.split[init_sequence_length:], 'actual')
    myplot(final_outputs, 'predicted')

    plt.legend(bbox_to_anchor=(.90, 1.05), loc=2, borderaxespad=0.)
    plt.savefig('sin_wave.png')
    plt.show()

    return final_outputs

def main():
    graph = SinGraph()

    points = graph.generate()
    split = points

    print(len(points))

    dataloder = DataLoader(MyDataset(split, SEQUENCE_LENGTH), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    rnn = MyRNN(hidden_size=HIDDEN_NEURONS, num_layers=1).to(DEVICE)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    train_model(rnn, dataloader=dataloder, loss_function=loss_function, optimizer=optimizer, epochs=NUM_EPOCHS)
    test_model(rnn, dataloder, init_sequence_length=SEQUENCE_LENGTH)


if __name__ == '__main__':
    main()