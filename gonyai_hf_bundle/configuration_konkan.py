from transformers import PretrainedConfig

class KonkanSmallConfig(PretrainedConfig):
    model_type = "konkangpt"

    def __init__(
        self,
        vocab_size=32000,
        d_model=768,
        n_layers=12,
        n_heads=12,
        d_ff=3072,
        max_len=1024,
        dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.dropout = dropout
        
        self.num_hidden_layers = n_layers
        self.hidden_size = d_model
        self.num_attention_heads = n_heads