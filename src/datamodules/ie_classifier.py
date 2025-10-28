import torch
from transformers import BertTokenizerFast
import jsonlines
import hydra


class IEClassifierDataset(torch.utils.data.Dataset):
    def __init__(self, path, max_length=512, tokenize=True, text_only=False):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased", do_lower_case=True)
        self.max_length = max_length
        self.tokenize = tokenize
        self.text_only = text_only
        self._load_data(path)

    def _load_data(self, path):
        self.texts = []
        self.labels = []
        with jsonlines.open(path, 'r') as f:
            for line in f:
                self.texts.append(line["text"])
                if "target" in line:
                    self.labels.append(line["target"])
        if self.tokenize:
            encodings = self.tokenizer(self.texts, truncation=True, padding=True, max_length=self.max_length)
            self.input_ids = encodings["input_ids"]
            self.attention_masks = encodings["attention_mask"]
            self.token_type_ids = encodings["token_type_ids"]

    def set_text_only_mode(self, mode:bool):
        self.text_only = mode

    def __getitem__(self, idx):
        if self.text_only:
            return {"text": self.texts[idx]}
        if self.tokenize:
            item = {"input_ids": torch.tensor(self.input_ids[idx]),
                    "token_type_ids": torch.tensor(self.token_type_ids[idx]),
                    "attention_mask": torch.tensor(self.attention_masks[idx])}
        else:
            item = self.tokenizer(self.texts[idx], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        if len(self.labels) > 0:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
