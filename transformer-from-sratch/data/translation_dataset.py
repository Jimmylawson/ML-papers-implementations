import torch
from torch.utils.data import Dataset


class TranslationDataset(Dataset):
    def __init__(self,dataset,src_tokenizer,tgt_tokenizer,max_len=64):
        self.dataset = dataset
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len

        self.src_bos = self.src_tokenizer.token_to_id("[BOS]")
        self.src_eos = self.src_tokenizer.token_to_id("[EOS]")
        self.tgt_bos = self.tgt_tokenizer.token_to_id("[BOS]")
        self.tgt_eos = self.tgt_tokenizer.token_to_id("[EOS]")
    def __len__(self):
        return len(self.dataset)
    #Call this when we need a particular example
    def __getitem__(self, index):
        example = self.dataset[index]

        english = example["translation"]["en"]
        french = example["translation"]["fr"]

        #Leave room for BOS and EOS
        src_ids = self.src_tokenizer.encode(english).ids[:self.max_len - 2]
        tgt_ids = self.tgt_tokenizer.encode(french).ids[:self.max_len - 2]
        #Complete source sequence
        src_tokens = [self.src_bos] + src_ids + [self.src_eos]

        #Decoder input and expected next token
        decoder_input = [self.tgt_bos] + tgt_ids
        expected_output = tgt_ids + [self.tgt_eos]

        return {
            "src_tokens":torch.tensor(src_tokens,dtype=torch.long),
            "decoder_input":torch.tensor(decoder_input,dtype=torch.long),
            "expected_output":torch.tensor(expected_output,dtype=torch.long)
        }


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    src_tokens = [item["src_tokens"] for item in batch]
    decoder_input = [item["decoder_input"] for item in batch]
    expected_output = [item["expected_output"] for item in batch]

    #Out tokenizers use [PAD] = 1
    pad_id = 1 #we choose Pad from our specialize tokenizer
    src_tokens = pad_sequence(src_tokens,
                              batch_first=True,
                              padding_value=pad_id)
    decoder_input = pad_sequence(decoder_input,
                               batch_first=True,
                               padding_value=pad_id)
    expected_output = pad_sequence(expected_output,batch_first=True,padding_value=pad_id)

    return {
        #(B,T) where B is batch size and T is max sequence length
        "src_tokens":src_tokens,
        "decoder_input":decoder_input,
        "expected_output":expected_output
    }