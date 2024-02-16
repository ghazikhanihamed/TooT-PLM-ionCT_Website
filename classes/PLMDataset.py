import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import re
from settings import settings
from torch.utils.data.dataloader import default_collate
from scipy import sparse


class PLMDataset(Dataset):
    def __init__(self, model, dataset, seqid_dict):
        self.model = model
        self.datasetFolderPath = settings.DATASET_PATH
        self.tokenizer = AutoTokenizer.from_pretrained(
            model["model"], do_lower_case=False
        )
        self.seqid_dict = seqid_dict

        self.seqs, self.labels = self.load_dataset(dataset)

        self.max_length = 1024

    def load_dataset(self, dataset):
        trainFilePath = os.path.join(self.datasetFolderPath, dataset)
        df = pd.read_csv(trainFilePath, names=["sequence", "label", "id"], skiprows=1)
        label_encoder = {
            "IC": settings.IONCHANNELS,
            "IT": settings.IONTRANSPORTERS,
            "MP": settings.MEMBRANE_PROTEINS,
        }
        # We extract the labels from the file name. The first and the second part split by _ are the labels based on the label encoder.
        labels = [
            label_encoder[dataset.split("_")[0]],
            label_encoder[dataset.split("_")[1]],
        ]
        # labels = [
        #     dataset.split("_")[0],
        #     (
        #         dataset.split("_")[1]
        #         if dataset.split("_")[1] != "membraneproteins"
        #         else "membrane_proteins"
        #     ),
        # ]
        df["labels"] = np.where(df["label"] == labels[0], 1, 0)

        seq = list(df["sequence"])
        label = list(df["labels"])

        assert len(seq) == len(label)

        return seq, label

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.model["name"] == "ProtBERT" or self.model["name"] == "ProtBERT-BFD":
            # We put a space between each amino acid
            seq = " ".join("".join(self.seqs[idx].split()))
        else:
            seq = "".join(self.seqs[idx].split())

        # Replace UZOB with X
        seq = re.sub(r"[UZOB]", "X", seq)

        seq_ids = self.tokenizer(seq, truncation=True, max_length=self.max_length)

        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}

        sample["labels"] = torch.tensor(self.labels[idx])

        return sample


class GridDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]
