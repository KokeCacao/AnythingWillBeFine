import torch
import torch.nn as nn


class LSTM(nn.Module):

    def __init__(
        self,
        sequence_length,
        input_size,
        output_size,
        num_embeddings,
        embedding_dim,
        batch_size,
        device,
        hidden_layer_size=128,
        num_layers=1,
        sparse_grad=False,
    ):
        super().__init__()
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(
            self.input_size, self.hidden_layer_size, num_layers, device=device
        )  # [sequence_length, batch_size, input_size] -> [sequence_length, batch_size, hidden_layer_size]
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim,
                                      padding_idx=None,
                                      sparse=sparse_grad,
                                      device=device
                                      )
        self.linear = nn.Linear(
            self.hidden_layer_size * self.sequence_length + embedding_dim,
            output_size, device=device
        )  # [N, hidden_layer_size * sequence_length + embedding_dim] -> [N, output_size]

    def clear_hidden_state(self, batch_size, device):
        self.hidden_cell = (torch.zeros(self.num_layers, batch_size,
                                        self.hidden_layer_size, device=device),
                            torch.zeros(self.num_layers, batch_size,
                                        self.hidden_layer_size, device=device))

    def forward(self, input_seq, stock_id):
        input_seq = input_seq.transpose(
            0, 1
        )  # [batch_size, sequence_length, input_size] -> [sequence_length, batch_size, input_size]
        input_seq = input_seq.reshape(input_seq.shape[0], -1, self.input_size)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)

        lstm_out = lstm_out.transpose(
            0, 1
        )  # [sequence_length, batch_size, hidden_layer_size] -> [batch_size, sequence_length, hidden_layer_size]
        lstm_out = lstm_out.reshape(
            lstm_out.shape[0], -1
        )  # [batch_size, sequence_length, hidden_layer_size] -> [batch_size, sequence_length * hidden_layer_size]

        predictions = self.linear(
            torch.cat((lstm_out, self.embedding(stock_id)), dim=-1)
        )  # [batch_size, sequence_length * hidden_layer_size + embedding_dim] -> [batch_size, output_size]
        return predictions  # [batch_size, output_size]
