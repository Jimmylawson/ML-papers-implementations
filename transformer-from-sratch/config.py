


def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "vocab_size" : 30000,
        "lr": 1e-4,
        "seq_len" : 350,
        "d_model": 512,
        "d_ff": 2048,
        "lang_src": "en",
        "lang_tgt":"it",
        "tokenizer_file": "tokenizer_{0}.json"
    }