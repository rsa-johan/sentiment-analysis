from lightning import LightningDataModule
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

import polars as pl
import torch

class SentimentDataLoader(LightningDataModule):
    def __init__(self, batch_size: int = 32, model_name="", device="") -> None:
        super().__init__()
        self.batch_size = batch_size
        self.embedder = SentenceTransformer(model_name, device=device)
        self.device = device

    def setup(self, stage: str) -> None:
        if stage == "fit":
            train_path = "dataset/twitter_training.csv"
            self.train_df = pl.read_csv(train_path, new_columns=["Sno", "Category", "Label", "Text"]).filter(pl.col("Label") != "Irrelevant").drop_nulls()
            text = self.train_df["Text"].to_list()
            label = self.train_df.with_columns(self.train_df["Label"].to_dummies()).drop("Label")
            train_y = label.select([x for x in label.columns if x.startswith("Label")]).to_torch()
            with torch.no_grad():
                train_x = torch.tensor(self.embedder.encode(text), dtype=torch.float32)
                self.train_data = TensorDataset(train_x.to(self.device), train_y.float().to(self.device))

        elif stage == "test":
            test_path = "dataset/twitter_training.csv"
            self.test_df = pl.read_csv(test_path, new_columns=["Sno", "Category", "Label", "Text"]).filter(pl.col("Label") != "Irrelevant").drop_nulls()
            text = self.test_df["Text"].to_list()
            label = self.test_df.with_columns(self.test_df["Label"].to_dummies()).drop("Label")
            test_y = label.select([x for x in label.columns if x.startswith("Label")]).to_torch()
            with torch.no_grad():
                test_x = torch.tensor(self.embedder.encode(text), dtype=torch.float32)
                self.test_data = TensorDataset(test_x.to(self.device), test_y.float().to(self.device))


    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
