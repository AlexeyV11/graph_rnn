import random
import itertools
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from graph_gen import SinGraph

class MyRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout_rate=0.05):
        super(MyRNN, self).__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size
        #
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=self.num_layers, dropout=dropout_rate)
        self.out = nn.Linear(hidden_size, 2)

    # input of shape (seq_len, batch, input_size):
    # h_0 of shape (num_layers * num_directions, batch, hidden_size):
    def forward(self, x, h_0):
        result, h_state = self.rnn(x, h_0)
        y_last = self.out(result[-1, :, :]) # last output
        return y_last, h_state

    def create_h0(self, batch_size):
        return torch.zeros([self.num_layers, batch_size, self.hidden_size]).to(DEVICE)


class MyDataset(Dataset):
    """PyTorch dataset class so we can use their Dataloaders."""

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
            output = self.split[idx + self.seq_length]

            return np.array(input, dtype='float32'), np.array(output, dtype='float32')



LEARNING_RATE = 0.01
BATCH_SIZE = 100
NUM_EPOCHS = 500
SEQUENCE_LENGTH = 50
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(model, dataloader, loss_function, optimizer, epochs):
    model.train()
    loss_all = []

    # Train loop.
    for epoch in range(epochs):
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.permute([1, 0, 2])

            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            h_state = model.create_h0(BATCH_SIZE)

            # Zero the gradients.
            optimizer.zero_grad()

            # Run our chosen rnn model.
            output, _ = model(x_batch, h_state)

            # Calculate loss.
            loss = loss_function(output, y_batch)

            # Backprop and perform update step.
            loss.backward()
            optimizer.step()

        loss_all.append(loss.cpu().data.numpy())
        print('train loss epoch{}: '.format(epoch), loss.cpu().data.numpy())

    torch.save(model.state_dict(), 'trained_rnn_model.pt')



def generate_predictions(model, dataloader, init_sequence_length):
    """From a trained model predict """
    model.eval()

    h_state = model.create_h0(1).to(DEVICE)  # Initial state is all zero.
    batch_input, batch_y = dataloader.dataset[0]

    initial_input = torch.Tensor(batch_input[:,np. newaxis, :]).to(DEVICE)\

    final_outputs = []
    for _ in range(len(dataloader.dataset)-init_sequence_length):

        output, _ = model(initial_input, h_state)
        final_outputs.append(output.cpu().data.squeeze_())

        # Pop off the first element of sequence then add on our latest generated point (use our predicted values in next predictions).
        initial_input.data[0:init_sequence_length-1, :, :] = initial_input.data[1:init_sequence_length, :, :]
        initial_input.data[init_sequence_length-1, :, :] = output.data

    def scatter(points, label_name):
        xx, yy = zip(*points)
        xx = list(map(float, xx))
        yy = list(map(float, yy))
        plt.plot(yy, label=label_name)

    scatter(dataloader.dataset.split[init_sequence_length:], 'actual')
    scatter(final_outputs, 'predicted')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('sin_wave.png')
    plt.show()

    return final_outputs

def main():
    graph = SinGraph()

    points = graph.generate()

    middle = int(len(points) * 0.8)

    split_train = points[0:middle]
    split_test = points[middle:]


    train_dataloder = DataLoader(MyDataset(split_train, SEQUENCE_LENGTH), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_dataloder = DataLoader(MyDataset(split_test, SEQUENCE_LENGTH), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    rnn = MyRNN(input_size=2, hidden_size=8, num_layers=1).to(DEVICE)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=LEARNING_RATE)
    loss_function = nn.MSELoss()

    train_model(rnn, dataloader=train_dataloder, loss_function=loss_function, optimizer=optimizer, epochs=NUM_EPOCHS)

    # FIXME: TODO:
    # trained data!!!
    generate_predictions(rnn, train_dataloder, init_sequence_length=SEQUENCE_LENGTH)


if __name__ == '__main__':
    main()