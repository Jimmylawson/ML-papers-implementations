from tokenizers import Tokenizer #manage the tokenization pipeline
from tokenizers.models import BPE #Defines the subword tokenization algorithm
from tokenizers.trainers import BpeTrainer #Learns BPE merges and vocabulary from text
from tokenizers.pre_tokenizers import Whitespace #split text into initial word/punctuation units


english_tokenizer = Tokenizer(
    BPE(unk_token="[UNK]") #UNK represent unknown token
)
french_tokenizer = Tokenizer(
    BPE(unk_token="[UNK]") #UNK represent unknown token
)

english_tokenizer.pre_tokenizer = Whitespace()
french_tokenizer.pre_tokenizer = Whitespace()

special_tokens = [
    "[UNK]", #represent unknown tokens
    "[PAD]", #pads shorter sentences to equal length within batch
    "[BOS]", #marks the beginning of a sentence
    "[EOS]" #Marks the end of a sentence
]

english_trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=special_tokens
)
french_trainer = BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=special_tokens
)


