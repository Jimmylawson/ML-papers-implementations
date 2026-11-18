import torch
import torch.nn as nn
from torch.cuda import device_of
from torch.utils.data import Dataset,DataLoader, random_split

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace #Split the word according to white space
from pathlib import Path
from config import get_config

from dataset import BilingualDataset

def get_all_sentences(ds,lang):
     for item in ds:
          # yield produce one sentence at a time rather than creating a list containing every sentence
          yield item["translation"][lang]

def get_ds(config):
     ds = load_dataset(
          "Helsinki-NLP/opus_books",
          f"{config['lang_src']}-{config['lang_tgt']}",
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

     def fits_sequence_length(example):
          translation = example["translation"]
          src_length = len(tokenizer_src.encode(translation[config["lang_src"]]).ids)
          tgt_length = len(tokenizer_tgt.encode(translation[config["lang_tgt"]]).ids)
          # Source needs SOS and EOS; decoder input/labels each need one.
          return (
               src_length + 2 <= config["seq_len"]
               and tgt_length + 1 <= config["seq_len"]
          )

     train_count, val_count = len(train_raw), len(val_raw)
     train_raw = [example for example in train_raw if fits_sequence_length(example)]
     val_raw = [example for example in val_raw if fits_sequence_length(example)]
     print(f"Training pairs removed: {train_count - len(train_raw)}")
     print(f"Validation pairs removed: {val_count - len(val_raw)}")

     if not train_raw or not val_raw:
          raise ValueError(
               "Sequence-length filtering left an empty training or validation split. "
               "Increase config['seq_len'] or use more data."
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

device = ""
if torch.cuda.is_available():
     device = torch.device("cuda")
elif torch.backends.mps.is_available():
          device = torch.device("mps")
else:
     device = torch.device("cpu")

def train_one_epoch(model,train_loader,optimizer,loss_fn,device):
     model.train()
     total_loss = 0.0

     for batch in train_loader:
          # 1. Move batch tensors to device
          encoder_input = batch["encoder_input"].to(device)
          decoder_input = batch["decoder_input"].to(device)
          encoder_mask = batch["encoder_mask"].to(device)
          decoder_mask = batch["decoder_mask"].to(device)
          label = batch["label"].to(device)

          #Clear old gradients from the previous batch
          optimizer.zero_grad()
          # 2. Run encoder forward pass
          encoder_output = model.encode(encoder_input,encoder_mask)
          # 3. Run decoder forward pass
          decoder_output = model.decode(encoder_output,encoder_mask,decoder_input,decoder_mask)
          # 4. Project decoder output to vocabulary logits
          #Example: (Batch x Sequence x Vocabulary) =>Before reshape
          # ouput.shape = (8, 128,30000)
          # 8 sentence, 128 positions per sentence
          #30,000 vocabulary scores per position
          output = model.project(decoder_output)
          #After reshaping => (1024, 30000) => Predictions x Vocabulary
          # 5. Calculate loss
          loss = loss_fn(
               output.reshape(-1, output.size(-1)),
               label.reshape(-1)
          )
          # 6. Backpropagate
          loss.backward()
          # 7. Update model weights
          optimizer.step()
          #Accumulate the loss
          total_loss += loss.item()

     return total_loss / len(train_loader)




from model.transformer  import build_transformer


def train_model(config):
     # Select device

     #2.Get DataLoader and tokenizers
     train_loader, val_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
     #3. Build  Transformers
     model = build_transformer(
          src_vocab_size=tokenizer_src.get_vocab_size(),
          tgt_vocab_size=tokenizer_tgt.get_vocab_size(),
          src_seq_len=config["seq_len"],
          tgt_seq_len=config["seq_len"],
          d_model=config["d_model"],
          N=config["N"],
          h=config["h"],
          dropout=config["dropout"],
          d_ff=config["d_ff"]
     ).to(device)

     #4 Create optimizer and loss function
     optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
     loss_fn = nn.CrossEntropyLoss(
          ignore_index=tokenizer_tgt.token_to_id("[PAD]")
     )
     # batch = next(iter(train_loader))
     #
     # encoder_input = batch["encoder_input"].to(device)
     # decoder_input = batch["decoder_input"].to(device)
     # encoder_mask = batch["encoder_mask"].to(device)
     # decoder_mask = batch["decoder_mask"].to(device)
     # label = batch["label"].to(device)
     #
     # print("Encoder:", encoder_input.shape)
     # print("Decoder:", decoder_input.shape)
     # print("Encoder mask:", encoder_mask.shape)
     # print("Decoder mask:", decoder_mask.shape)
     #
     # encoder_output = model.encode(encoder_input, encoder_mask)
     #
     # decoder_output = model.decode(
     #      encoder_output,
     #      encoder_mask,
     #      decoder_input,
     #      decoder_mask
     # )
     #
     # output = model.project(decoder_output)
     #
     # print("Output:", output.shape)
     # loss = loss_fn(
     #      output.reshape(-1, output.size(-1)),
     #      label.reshape(-1)
     # )
     # print("Loss:", loss.item())
     # print("Finite loss:", torch.isfinite(loss).item())
     for epoch in range(config["num_epochs"]):
         train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn,device)
         val_loss = validation_model(model,val_loader,loss_fn,device)
         print(
              f"Epoch {epoch + 1}/{config['num_epochs']} "
              f"| Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f}"
         )
     return model






def validation_model(model, val_dataloader, loss_fn, device):
    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)
            decoder_mask = batch["decoder_mask"].to(device)
            label = batch["label"].to(device)

            # Encoder forward pass
            encoder_output = model.encode(encoder_input, encoder_mask)

            # Decoder forward pass
            decoder_output = model.decode(
                encoder_output,
                encoder_mask,
                decoder_input,
                decoder_mask
            )

            # Projection
            output = model.project(decoder_output)

            # Calculate loss
            loss = loss_fn(
                output.reshape(-1, output.size(-1)),
                label.reshape(-1)
            )

            total_val_loss += loss.item()

    return total_val_loss / len(val_dataloader)



def get_or_build_tokenizer(config,ds,lang):
          tokenizer_path = Path(config["tokenizer_file"].format(lang))
          if tokenizer_path.exists():
               tokenizer =Tokenizer.from_file(str(tokenizer_path))
          else:
               trainer = WordLevelTrainer(
                    vocab_size=config["vocab_size"],
                    special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"]
               )
               tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
               tokenizer.pre_tokenizer = Whitespace()
               tokenizer.train_from_iterator(get_all_sentences(ds,lang), trainer=trainer)
               tokenizer.save(str(tokenizer_path))

          return tokenizer






