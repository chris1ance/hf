# Set up your API Key

In the terminal:

```
python
from huggingface_hub import login
login()
```

Enter your API key.

Choose `n` to the question `Add token as git credential?`

Then your token will be saved to `.cache/huggingface/token` in your home directory on Linux.

Doing this will allow you to download gated models after getting approval on the model page on huggingface.co.

# Installation

To install the latest branch of `transformers` directly from Github: `pip install git+https://github.com/huggingface/transformers`

# Downloading a Model

To download a model, you can just load it:

```
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=False,      
    use_safetensors=True,
)
```

# Loading a Model

## Reproducibility

Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`:

```
from transformers import set_seed
set_seed(42)
```

## AutoTokenizer

We load a tokenizer as follows:

```
from transformers import AutoTokenizer

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
```

`AutoTokenizer.from_pretrained` accepts several optional args. The following are optional, with suitable default values:

- `trust_remote_code (default: False)`

- `use_fast (default: True)`:  Use a fast Rust-based tokenizer if it is supported for a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead. The “Fast” implementations allows:

    * a significant speed-up in particular when doing batched tokenization and

    * additional methods to map between the original string (character and words) and the token space (e.g. getting the index of the token comprising a given character or the span of characters corresponding to a given token).


References:
- `AutoTokenizer`: https://huggingface.co/docs/transformers/model_doc/auto

## AutoModel

We load a model as follows:

```
from transformers import AutoModelForCausalLM

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=False,                  # Should always be set to False
    use_safetensors=True,                     # Should always be set to True
    torch_dtype="auto",                       # Other choices are torch.float16, torch.float32, or torch.bfloat16
    attn_implementation="flash_attention_2"   # Not supported by all models
    )

model.eval()   # Always put model in eval mode for inference
```

FlashAttention-2 can be combined with other optimization techniques like quantization to further speedup inference. For example, you can combine FlashAttention-2 with 8-bit or 4-bit quantization:

```
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",                                          # have accelerate compute the most optimized device_map automatically
    trust_remote_code=False,                                    # Should always be set to False
    use_safetensors=True,                                       # Should always be set to True
    torch_dtype="auto",                                         # Other choices are torch.float16, torch.float32, or torch.bfloat16
    attn_implementation="flash_attention_2",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)   # or, BitsAndBytesConfig(load_in_4bit=True)
    )
```

Read about Flash-Attentention2 at: https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one

Key details:
* FlashAttention-2 can only be used when the model’s dtype is fp16 or bf16.

* FlashAttention-2 can be combined with other optimization techniques like quantization to further speedup inference. For example, you can combine FlashAttention-2 with 8-bit or 4-bit quantization.

* FlashAttention-2 does not support computing attention scores with padding tokens, you must manually pad/unpad the attention scores for batched inference when the sequence contains padding tokens. This leads to a significant slowdown for batched generations with padding tokens.