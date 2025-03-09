from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.nn import LSTM, BCEWithLogitsLoss, Dropout, Linear
from torch.optim import Adam
from lightning import LightningModule

class SentimentClassifier(LightningModule):
    def __init__(self, hidden_size, num_layers, output_size, bidirectional=False) -> None:
        super().__init__()
        self.lstm_layer = LSTM(input_size=384, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)
        self.fc = Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = Dropout(0.3)
        self.loss = BCEWithLogitsLoss()

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        print(f"Batch {batch_idx}")
        input_i, label_i = batch
        output_i = self.forward(input_i)
        loss_i = self.loss(output_i, label_i)
        self.log('train_loss', loss_i)
        return loss_i

    def forward(self, lstm_input):
        lstm_out, _ = self.lstm_layer(lstm_input)
        final_out = self.fc(self.dropout(lstm_out))
        return final_out
