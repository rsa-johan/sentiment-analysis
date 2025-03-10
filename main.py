from sentence_transformers import SentenceTransformer
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

from dataloader import SentimentDataLoader
from model import SentimentClassifier

embedder_model = "sentence-transformers/all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

logger = TensorBoardLogger("tb_logs", name="LSTM_model")
model = SentimentClassifier(hidden_size=512, num_layers=2, output_size=3, bidirectional=True)
#model.load_state_dict(torch.load("model.pt", weights_only=True))

trainer = L.Trainer(limit_train_batches=0.5, max_epochs=10, accelerator="gpu", devices="auto", default_root_dir="models/", logger=logger)

dataloader = SentimentDataLoader(model_name=embedder_model, device=device)
trainer.fit(model=model, datamodule=dataloader)

torch.save(model.state_dict(), "model.pt")
trainer.test(model=model, datamodule=dataloader)

with torch.inference_mode():
    output = model(torch.tensor(SentenceTransformer(embedder_model, device=device).encode(["I am happy", "I am not happy"]), dtype=torch.float32))
    print(output)
    prediction = torch.argmax(output, dim=1)
    print(prediction)
