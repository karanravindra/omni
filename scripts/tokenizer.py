from datasets import load_dataset
from tokenizers import (
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)

vocab_size = 2048

d = load_dataset("noanabeshima/TinyStoriesV2", split="train")


def batch_iterator(batch_size=100):
    tok_dataset = d.select_columns("text")
    for batch in tok_dataset.iter(batch_size):
        yield batch["text"]


tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.NFKD()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.post_processor = processors.Sequence(
    [
        processors.TemplateProcessing(
            single="[SOS] $A [EOS]",
            special_tokens=[
                ("[SOS]", 2),
                ("[EOS]", 3),
            ],
        ),
        processors.ByteLevel(),
    ]
)


trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
)


tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(d))
tokenizer.save(f"tokenizer-{vocab_size}.json")
