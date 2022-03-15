import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset


class MLDataset(Dataset):
    def __init__(self, df):
        self.pandas_df = df
        self.user_id_numpy = (df["userID"].astype("int64") - 1).to_numpy()
        self.item_id_numpy = df["itemID"].astype("int64").to_numpy()
        self.rating_numpy = (df["rating"].astype(bool).astype("f")).to_numpy()

        self.user_id = torch.tensor((df["userID"].astype("int64") - 1).to_numpy())
        self.item_id = torch.tensor(df["itemID"].astype("int64").to_numpy())
        self.rating = torch.tensor(
            (df["rating"].astype(bool).astype("f")).to_numpy(), requires_grad=True
        )

    def __len__(self):
        return len(self.pandas_df)

    def __getitem__(self, idx):

        # row = self.pandas_df.iloc[idx]
        # user_id = row["userID"].astype("int64")
        # item_id = row["itemID"].astype("int64")
        # rating = row["rating"].astype("f")
        user_id = self.user_id_numpy[idx]
        item_id = self.item_id_numpy[idx]
        rating = self.rating_numpy[idx]

        return user_id, item_id, rating
