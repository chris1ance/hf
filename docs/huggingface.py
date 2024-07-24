# +
# ruff: noqa
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

set_seed(42)  # Set seed or reproducibility
# -

model_name = "google/gemma-2-9b-it"  # "stabilityai/stablelm-zephyr-3b"

# # Releasing GPU Memory
#
# Let’s define a `flush(...)` function to free all allocated memory
#
# ```
# del pipe
# del model
#
# import gc
# import torch
#
# def flush():
#   gc.collect()
#   torch.cuda.empty_cache()
#   torch.cuda.reset_peak_memory_stats()
# ```
#
# In the recent version of the accelerate library, you can also use an utility method called `release_memory()`:
#
# ```
# from accelerate.utils import release_memory
# # ...
# release_memory(model)
# ```
#
# Reference: https://huggingface.co/docs/transformers/en/llm_tutorial_optimization

# # Key Docs
#
# * `AutoTokenizer` and `AutoModel`: https://huggingface.co/docs/transformers/model_doc/auto
# * `Tokenizer`: https://huggingface.co/docs/transformers/en/main_classes/tokenizer

# # Tokenizers

# We load a tokenizer as follows:

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

# Tokenizers have several attributes:

dir(tokenizer)

# Note: The context length according to the tokenizer may differ from
# the context length according to the model
max_length = tokenizer.model_max_length
print(f"The maximum context length for the model '{model_name}' is: {max_length}")

# Some attributes may have to be set manually:

if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

# ## Tokenizer without chat template

# We can visualize how the tokenizer tokenizer text:

text = "We are happy to show you the Transformers library."
tokenizer.tokenize(text)

# Alternatively, if you want to see the token IDs instead of the token strings, you can use the `tokenizer.encode` method, which converts the text into a list of token IDs:

token_ids = tokenizer.encode(text)
print(token_ids)

# +
# Note: Putting the text in a list leads to both of the following lines failing
# tokenizer.tokenize([text]),
# tokenizer.encode([text])
# -

# `tokenizer(text, return_tensors="pt")` takes a string or a list of strings as input and returns a dictionary of tensors.
#
# The `return_tensors` parameter specifies the type of tensors to return. In this case, "pt" stands for PyTorch tensors.
#
# The returned dictionary contains keys "input_ids", "attention_mask", and "token_type_ids" (depending on the tokenizer), and the corresponding values are PyTorch tensors.
# * The "input_ids" tensor represents the token IDs, similar to the output of tokenizer.encode.
# * The "attention_mask" tensor indicates which tokens should be attended to during the model's computation.
# * The "token_type_ids" tensor is used for sequence classification tasks to distinguish between different segments of the input.

# +
encoding = tokenizer(text, return_tensors="pt")

print(encoding)
# -

# Note: Putting the text in a list leads to the same output here
encoding = tokenizer([text], return_tensors="pt")
print(encoding)

# Batched inputs are often different lengths, so they can’t be converted to fixed-size tensors. Padding and truncation are strategies for dealing with this problem, to create rectangular tensors from batches of varying lengths.
#
# * `padding=True`: pad to the longest sequence in the batch (no padding is applied if you only provide a single sequence). The padding token is `tokenizer.pad_token`.
#
# * `truncation=True`: truncate to a maximum length specified by the `max_length` argument or the maximum length accepted by the model if no `max_length` is provided (If `max_length` is left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters.). This will truncate token by token, removing a token from the longest sequence in the pair until the proper length is reached.

# +
batch_sentences = [
    "A list of colors: red, blue",
    "Portugal is",
]

encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt")

print(encoded_inputs)
# -

# ## Tokenizer with chat templates

# +
# Note: Putting the dict in a list is required!
prompt = [{"role": "user", "content": text}]

[
    tokenizer.apply_chat_template(prompt, tokenize=False),
    # When using chat templates as input for model generation, use add_generation_prompt=True to add a generation prompt.
    # This ensures that when the model generates text it will write a bot response instead of doing something unexpected
    tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False),
    # This will return the input_ids for generation
    tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ),
    # return_dict = True will return the attention mask as well
    tokenizer.apply_chat_template(
        prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
    ),
]

# +
longer_prompt = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]

[
    tokenizer.apply_chat_template(longer_prompt, tokenize=False),
    # When using chat templates as input for model generation, use add_generation_prompt=True to add a generation prompt.
    # This ensures that when the model generates text it will write a bot response instead of doing something unexpected
    tokenizer.apply_chat_template(longer_prompt, add_generation_prompt=True, tokenize=False),
    # This will return the input_ids for generation
    tokenizer.apply_chat_template(
        longer_prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ),
    # return_dict = True will return the attention mask as well
    tokenizer.apply_chat_template(
        longer_prompt,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ),
]
# -

# Starting with transformers v4.40, batch chat templating is supported. See https://github.com/huggingface/transformers/pull/29222 for more information.

# +
batch_prompts = [
    [{"role": "user", "content": text}],
    [{"role": "user", "content": "How are you?"}],
]

tokenizer.apply_chat_template(
    batch_prompts,
    padding=True,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    return_dict=True,
)
# -

# ## Key Params
#
# * **text** (str, List[str] or List[int]) — The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the tokenize method) or a list of integers (tokenized string ids using the convert_tokens_to_ids method).
#
# * **text_pair** (str, List[str] or List[int], optional) — Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using the tokenize method) or a list of integers (tokenized string ids using the convert_tokens_to_ids method).
#
# * **add_special_tokens** (bool, optional, defaults to True) — Whether or not to add special tokens when encoding the sequences. This will use the underlying PretrainedTokenizerBase.build_inputs_with_special_tokens function, which defines which tokens are automatically added to the input ids. This is usefull if you want to add bos or eos tokens automatically.
#
# * **padding** (bool, str or PaddingStrategy, optional, defaults to False) — Activates and controls padding. Accepts the following values:
#
#     True or 'longest': Pad to the longest sequence in the batch (or no padding if only a single sequence if provided).
#
#     'max_length': Pad to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided.
#
#     False or 'do_not_pad' (default): No padding (i.e., can output a batch with sequences of different lengths).
#
# * **truncation** (bool, str or TruncationStrategy, optional, defaults to False) — Activates and controls truncation. Accepts the following values:
#
#     True or 'longest_first': Truncate to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided.
#
#     'only_first': Truncate to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
#
#     'only_second': Truncate to a maximum length specified with the argument max_length or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided.
#
#     False or 'do_not_truncate' (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).
#
# * **max_length** (int, optional) — Controls the maximum length to use by one of the truncation/padding parameters.
#
#     If left unset or set to None, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.
#
# * **stride** (int, optional, defaults to 0) — If set to a number along with max_length, the overflowing tokens returned when return_overflowing_tokens=True will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.
#
# * **is_split_into_words** (bool, optional, defaults to False) — Whether or not the input is already pre-tokenized (e.g., split into words). If set to True, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.
#
# * **pad_to_multiple_of** (int, optional) — If set will pad the sequence to a multiple of the provided value. Requires padding to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
#
# * **return_tensors** (str or TensorType, optional) — If set, will return tensors instead of list of python integers. Acceptable values are:
#
#     'tf': Return TensorFlow tf.constant objects.
#     'pt': Return PyTorch torch.Tensor objects.
#     'np': Return Numpy np.ndarray objects.
#
# * ****kwargs** — Passed along to the .tokenize() method.

# ## Links
#
# * Padding and Truncation:
#     - https://huggingface.co/docs/transformers/en/pad_truncation
#
# * EOS token as padding token:
#     - https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954/9

# ## FAQ
#
# ### Padding decoder-only LLMs on the left or right?
#
# The left.
#
# Reference: https://huggingface.co/docs/transformers/llm_tutorial#wrong-padding-side
#
# ### Sometimes a huggingface auto class tokenizer returns token_type_ids and sometimes it does not. Why?
#
# The presence or absence of `token_type_ids` in the output of a Hugging Face tokenizer depends on the specific model architecture and the tokenizer's configuration.
#
# `token_type_ids` are used to distinguish different segments or sequences within the input. They are commonly used in tasks that involve multiple segments, such as sequence pair classification or question answering, where there is a need to differentiate between the segments.
#
# Here are a few reasons why `token_type_ids` may or may not be returned by a tokenizer:
#
# 1. Model Architecture:
#   - Some model architectures, such as BERT and its variants (e.g., RoBERTa, DistilBERT), utilize `token_type_ids` to differentiate between segments in the input.
#   - In these models, `token_type_ids` are used to indicate whether a token belongs to the first segment (e.g., sentence A) or the second segment (e.g., sentence B).
#   - The tokenizer for these models will typically return `token_type_ids` by default.
#
# 2. Single Sequence Models:
#   - Models like GPT-2, GPT-Neo, and XLNet are designed to process a single continuous sequence of tokens.
#   - These models do not require `token_type_ids` because they do not differentiate between segments.
#   - The tokenizers for these models usually do not return `token_type_ids` by default.
#
# 3. Tokenizer Configuration:
#   - Some tokenizers have a configuration option to control whether to return `token_type_ids` or not.
#   - For example, the `return_token_type_ids` parameter in the `encode_plus` or `__call__` method of the tokenizer can be set to `True` or `False` to explicitly specify whether to include `token_type_ids` in the output.
#
# 4. Task-specific Requirements:
#   - Depending on the specific task or downstream application, you may or may not need `token_type_ids`.
#   - For tasks that involve multiple segments or require differentiation between segments, you would typically want to include `token_type_ids`.
#   - For tasks that only deal with a single continuous sequence, `token_type_ids` may not be necessary.
#
# It's important to refer to the documentation or source code of the specific tokenizer you are using to understand its behavior and configuration options regarding `token_type_ids`.
#
# If you require `token_type_ids` for your task but the tokenizer does not return them by default, you can usually set the appropriate configuration option (e.g., `return_token_type_ids=True`) when calling the tokenizer to include them in the output.

# # Models

# +
import torch
from transformers import BitsAndBytesConfig

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    device_map="auto",
    trust_remote_code=False,  # Should always be set to False
    use_safetensors=True,  # Should always be set to True
    torch_dtype="auto",  # Other choices are torch.float16, torch.float32, or torch.bfloat16
)
# -

# n_ctx: The maximum context length (sequence length) the model can handle.
# n_embd: The dimensionality of the model's embeddings
# n_head: The number of attention heads in each layer.
# n_layer: The number of transformer layers in the model.
# vocab_size: The size of the model's vocabulary
config = model.config
print(config)

# +
max_context_length = model.config.max_position_embeddings
print(f"The maximum context length for the model '{model_name}' is: {max_context_length}")

model_device = model.device
print(f"The model '{model_name}' device is: {model_device}")

model_dtype = model.dtype
print(f"The model '{model_name}' dtype is: {model_dtype}")
# -

# # Generation

# +
prompt = [{"role": "user", "content": "How are you doing today?"}]

model_inputs = tokenizer.apply_chat_template(
    prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
).to(model_device)

input_length = model_inputs["input_ids"].shape[1]
print(f"Number of tokens in the prompt: {input_length}")
# -

# Convert input_ids back to text
decoded_text = tokenizer.decode(model_inputs["input_ids"][0])
decoded_text

# +
# prompt_lookup_num_tokens ref: https://twitter.com/joao_gante/status/1747322413006643259
prompt_lookup_num_tokens = 2 if model_device == "cpu" else 10

# By default, max_length = 20; set max_new_tokens to change
# use_cache ref: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958/5
with torch.no_grad():
    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_context_length - model_inputs["input_ids"].shape[1],
            use_cache=True,
            prompt_lookup_num_tokens=prompt_lookup_num_tokens,
        )
    except:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_context_length - model_inputs["input_ids"].shape[1],
            use_cache=True,
        )

generated_ids

# +
token_output = generated_ids[0]

[  # Basic decoding
    tokenizer.decode(token_output).strip(),
    # Remove special tokens (e.g. pad, EOS, BOS tokens)
    tokenizer.decode(token_output, skip_special_tokens=True).strip(),
    # clean_up_tokenization_spaces ref: https://discuss.huggingface.co/t/what-does-the-parameter-clean-up-tokenization-spaces-do-in-the-tokenizer-decode-function/17399/2
    tokenizer.decode(
        token_output, skip_special_tokens=True, clean_up_tokenization_spaces=True
    ).strip(),
]
# -

# ## Streaming
#
# See: https://huggingface.co/docs/transformers/generation_strategies

# +
from transformers import TextStreamer

prompt = [{"role": "user", "content": "Could you write me a sonnet about balloon monsters?"}]

model_inputs = tokenizer.apply_chat_template(
    prompt, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True
).to(model_device)

streamer = TextStreamer(tokenizer, skip_special_tokens=True)

with torch.no_grad():
    try:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_context_length - model_inputs["input_ids"].shape[1],
            use_cache=True,
            prompt_lookup_num_tokens=prompt_lookup_num_tokens,
            streamer=streamer,
        )
    except:
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_context_length - model_inputs["input_ids"].shape[1],
            use_cache=True,
            streamer=streamer,
        )
# -

generated_ids

token_output = generated_ids[0]
tokenizer.decode(token_output).strip()

# +
# skip_prompt (bool, optional, defaults to False) — Whether to skip the prompt to .generate() or not. Useful e.g. for chatbots.
# However, the prompt is still returned in the generated_ids
streamer = TextStreamer(tokenizer, skip_special_tokens=True, skip_prompt=True)

with torch.no_grad():
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_context_length - model_inputs["input_ids"].shape[1],
        use_cache=True,
        prompt_lookup_num_tokens=prompt_lookup_num_tokens,
        streamer=streamer,
    )
# -

generated_ids

token_output = generated_ids[0]
tokenizer.decode(token_output).strip()

# ## Decoding Strategies
#
# A decoding strategy for a model is defined in its generation configuration. When you load a model explicitly, you can inspect the generation configuration that comes with it through `model.generation_config`:

model.generation_config

# The default generation configuration limits the size of the output combined with the input prompt to a maximum of 20 tokens to avoid running into resource limitations. The default decoding strategy is greedy search, which is the simplest decoding strategy that picks a token with the highest probability as the next token.

# ## FAQ
#
# ## model(**input) vs. model.generate(**input)
#
# 1. `model(**input)`:
#    - When you call `model(**input)`, you are performing a forward pass through the model using the provided input.
#    - The `input` typically includes the tokenized input sequence, attention masks, and any other required inputs for the specific model.
#    - The model processes the input and returns the logits for each token in the input sequence.
#    - The logits represent the model's predicted probability distribution over the vocabulary for the next token at each position.
#    - However, `model(**input)` does not generate any new tokens or perform text generation.
#
# 2. `model.generate(**input)`:
#    - The `generate` method is specifically designed for text generation tasks. You provide the model with a prompt or a partial input sequence, and the model generates a continuation or completion of the prompt.
#    - The `generate` method takes care of the generation process internally, including decoding strategies like greedy search.
#    - You can control various aspects of the generation process using the arguments provided to `generate`, such as the maximum length, number of beams, temperature, top-k sampling, etc.
#    - The `generate` method returns the generated text or a list of generated sequences based on the specified generation parameters.
#    - Under the hood, `generate` repeatedly calls the model's forward pass to obtain the logits for each generated token and uses the decoding strategy to select the next token.
#
# It's important to note that not all models support the `generate` method. The `generate` method is typically available for autoregressive language models like GPT-2, GPT-Neo, and XLNet, which are designed for text generation tasks. Other models, such as BERT or RoBERTa, do not have a `generate` method because they are primarily used for language understanding tasks.

# ### torch.no_grad
#
# When performing text generation, it is recommended to use a PyTorch no_grad context manager. The no_grad context manager disables gradient computation during the generation process, which can provide several benefits:
#
# * Improved Performance: By disabling gradient computation, the no_grad context manager eliminates the overhead of tracking gradients during the generation process. This can lead to faster generation speed and reduced memory usage, especially when generating long sequences or handling large batches.
#
# * Reduced Memory Usage: When gradients are not needed, such as during inference or generation, disabling gradient computation with no_grad can significantly reduce the memory footprint.
#
# * Cleaner Code: Using the no_grad context manager makes it explicit that gradients are not required for the generation process. It serves as a clear indication that the code block is focused on inference or generation rather than training.
#
# Note that when using `model.generate()`, gradients are not computed by default, even without explicitly using no_grad. However, it is still a good practice to use no_grad for clarity and to ensure that gradients are not accidentally computed in case the generation process is combined with other operations that may require gradients.
#
