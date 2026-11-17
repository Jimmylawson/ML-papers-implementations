import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace #Split the word according to white space
from pathlib import Path
from config import get_config

config = get_config()

from dataset import BilingualDataset

trainer = WordLevelTrainer(
     vocab_size=config["vocab_size"],
     special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"]
)
def get_all_sentences(ds,lang):
     for item in ds:
          # yield produce one sentence at a time rather than creating a list containing every sentence
          yield item["translation"][lang]

def get_ds(config):
     ds = load_dataset(
          "Helsinki-NLP/opus_books",
          "en-it",
          split="train"
     )
     train_size = int(0.9 * len(ds))
     val_size = len(ds) - train_size
     train_raw, val_raw = random_split(ds, [train_size, val_size], generator=torch.Generator().manual_seed(42))

     tokenizer_src = get_or_build_tokenizer(
          config, train_raw, config["lang_src"]
     )
     tokenizer_tgt = get_or_build_tokenizer(
          config, train_raw, config["lang_tgt"]
     )

     #Converts the  those sentences into the token IDs,padding,labels, and attention
     #Maks your Transformer needs

     train_ds = BilingualDataset(
          train_raw,
          tokenizer_src,
          tokenizer_tgt,
          config["lang_src"],
          config["lang_tgt"],
          config["seq_len"]
     )

     val_ds  = BilingualDataset(
          val_raw,
          tokenizer_src,
          tokenizer_tgt,
          config["lang_src"],
          config["lang_tgt"],
          config["seq_len"]
     )
     train_dataloader = DataLoader(
          train_ds,
          batch_size=config["batch_size"],
          shuffle=True,
     )
     val_dataloader = DataLoader(
          val_ds,
          batch_size=config["batch_size"],
          shuffle=False
     )

     return train_dataloader,val_dataloader, tokenizer_src, tokenizer_tgt

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

def train_model(model,train_loader,loss_fn,device):
     pass

def validation_model(model, val_dataloader,loss_fn, device):
     model.eval()
     with torch.no_grad():
          for batch in val_dataloader:
               pass


def get_or_build_tokenizer(config,ds,lang):
          tokenizer_path = Path(config["tokenizer_file"].format(lang))
          if tokenizer_path.exists():
               tokenizer =Tokenizer.from_file(str(tokenizer_path))
          else:
               tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
               tokenizer.pre_tokenizer = Whitespace()
               tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
               tokenizer.save(str(tokenizer_path))

          return tokenizer






