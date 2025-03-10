from lightning.pytorch.utilities.types import OptimizerLRScheduler
from lightning import LightningModule

import torch
import torch.functional as F
from torch.nn import LSTM, BCEWithLogitsLoss, Dropout, Linear, ReLU
from torch.nn.modules import CrossEntropyLoss
from torch.optim import Adam
from torchmetrics import Accuracy

class SentimentClassifier(LightningModule):
    def __init__(self, hidden_size, num_layers, output_size, bidirectional=False) -> None:
        super().__init__()
        self.lstm_layer = LSTM(input_size=384, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.fc1 = Linear(hidden_size * (2 if bidirectional else 1), hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size//2)
        self.fc3 = Linear(hidden_size//2, output_size)
        self.dropout = Dropout(0.3)
        self.relu = ReLU()
        self.loss = CrossEntropyLoss()
        self.accuracy = Accuracy(task="multiclass", num_classes=3)

    def forward(self, lstm_input):
        lstm_out, _ = self.lstm_layer(lstm_input)
        final_out = self.fc1(self.dropout(lstm_out))
        final_out = self.relu(final_out)
        final_out = self.fc2(final_out)
        final_out = self.relu(final_out)
        final_out = self.fc3(final_out)
        final_out = self.relu(final_out)
        return final_out

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self(input_i)
        loss_i = self.loss(output_i, label_i)
        self.log('train_loss', loss_i, prog_bar=True)
        return loss_i

    def test_step(self, batch, batch_idx):
        input_i, label_i = batch
        output_i = self(input_i)
        loss_i = self.loss(output_i, label_i)
        self.log('test_loss', loss_i, prog_bar=True)
        return loss_i
