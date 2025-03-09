import lightning as L

from dataloader import SentimentDataLoader
from model import SentimentClassifier
import torch

model = SentimentClassifier(hidden_size=512, num_layers=2, output_size=3, bidirectional=True)

trainer = L.Trainer(limit_train_batches=100, max_epochs=2, accelerator="gpu", devices="auto", default_root_dir="models/")

train_dataloader = SentimentDataLoader()

trainer.fit(model=model, train_dataloaders=train_dataloader)

torch.save(model, "model.pt")
