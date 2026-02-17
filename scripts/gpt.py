from datasets import load_dataset
from lightning import LightningDataModule
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    trainers,
)
from torch.utils.data import DataLoader


class TinyStoriesV2(LightningDataModule):
    def __init__(self, batch_size=1, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        data = load_dataset("noanabeshima/TinyStoriesV2")

        self.dataset = data

    def train_dataloader(self):
        data = self.dataset["train"].with_format("torch")
        return DataLoader(
            data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        data = self.dataset["validation"].with_format("torch")
        return DataLoader(
            data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        data = self.dataset["validation"].with_format("torch")
        return DataLoader(
            data, batch_size=self.batch_size, num_workers=self.num_workers
        )


dm = TinyStoriesV2()
dm.setup()

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.UnigramTrainer(
    vocab_size=8192,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["<PAD>", "<BOS>", "<EOS>"],
)


def batch_iterator(batch_size=1000):
    # Only keep the text column to avoid decoding the rest of the columns unnecessarily
    tok_dataset = dm.dataset["train"].select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]


tokenizer.train_from_iterator(
    batch_iterator(), trainer=trainer, length=len(dm.dataset["train"])
)

tokenizer.save("tokenizer.json")
