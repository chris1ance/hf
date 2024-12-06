"""Module provides classes for interacting with and managing Language Models (LLMs) for text generation tasks.

The module is structured around the concept of a base language model class (`BaseLLM`) and its specialized
subclasses (`LLM`) that are tailored for specific types of language models. These classes
facilitate the loading, configuration, and operation of various language models, offering functionalities
for generating text responses, handling conversation history, and managing model-specific settings.

Classes:
    BaseLLM: A base class for language models. It provides foundational functionalities such as managing
    chat history and system prompts. This class is designed to be extended by more specific LLM classes
    that handle different types of models.

    LLM: A subclass of `BaseLLM`, designed to handle general language models available
    through the Hugging Face Transformers library. It includes additional functionalities specific to these
    types of models, such as handling model quantization and providing interfaces for generating text responses.
"""

import copy
import gc
from typing import TypedDict

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

########################################################################################


# Type alias for a conversation entry
class ConversationEntry(TypedDict):
    """Type alias for a conversation entry.

    Example 1 (role = user):
    {"role": "system", "content": "You are a helpful assistant."}

    Example 2 (role = user):
    {"role": "user", "content": "Hello, how are you?"}

    Example 3 (role = assistant):
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
    """

    role: str  # One of: "system" | "user" | "assistant"
    content: str  # The actual message content


def validate_conversation_entry(entry: ConversationEntry) -> None:
    """Validate that an entry matches the ConversationEntry type specification.

    Raises:
        TypeError: If entry is not a dict or has wrong value types
        KeyError: If entry is missing required keys
        ValueError: If role or content values are invalid

    """
    # Check if entry is a dictionary
    if not isinstance(entry, dict):
        raise TypeError(f"Entry must be a dictionary, got {type(entry)}")

    # Check for required keys
    if "role" not in entry:
        raise KeyError("Entry missing required 'role' field")
    if "content" not in entry:
        raise KeyError("Entry missing required 'content' field")

    # Check for extra keys
    extra_keys = set(entry.keys()) - {"role", "content"}
    if extra_keys:
        raise ValueError(f"Entry contains unexpected fields: {extra_keys}")

    # Check value types
    if not isinstance(entry["role"], str):
        raise TypeError(f"'role' must be a string, got {type(entry['role'])}")
    if not isinstance(entry["content"], str):
        raise TypeError(f"'content' must be a string, got {type(entry['content'])}")

    # Validate role value
    valid_roles = {"system", "user", "assistant"}
    if entry["role"] not in valid_roles:
        raise ValueError(f"'role' must be one of {valid_roles}, got '{entry['role']}'")

    # Check content is not empty
    if not entry["content"].strip():
        raise ValueError("'content' cannot be empty or only whitespace")


# Type alias for a conversation, which is a list of conversation entries
# Example:
# conversation = [
#    {"role": "system", "content": "You are a helpful assistant."},
#    {"role": "user", "content": "Hello, how are you?"},
#    {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
#    {"role": "user", "content": "What's the weather like?"},
#    {"role": "assistant", "content": "I don't have access to current weather information."}
# ]
Conversation = list[ConversationEntry]

########################################################################################


class LLM:
    """A class representing a Language Model (LLM) for generating text responses, derived from the BaseLLM class.

    This class provides a comprehensive interface for interacting with large language models,
    handling everything from model initialization to response generation. It supports various
    features including chat history management, streaming outputs, and batch processing.

    Attributes:
    ----------
    author : str | None
        Author or organization that created the model, extracted from model name
    model_name : str
        Complete name/path of the model being used
    tokenizer : AutoTokenizer
        HuggingFace tokenizer instance for the model
    model : AutoModelForCausalLM
        The loaded language model instance
    device : torch.device | None
        Device where the model is loaded (GPU/CPU)
    config : AutoConfig
        Model configuration parameters
    max_position_embeddings : int
        Maximum sequence length supported by the model
    chat_history : ChatHistory
        List of tuples containing (prompt, response) pairs
    system_prompt : str | None
        Default system prompt used for all interactions

    Example:
    -------
    >>> # Initialize the model
    >>> model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    >>> model = LLM(model_id)
    >>>
    >>> # Single query
    >>> response = model.query_llm("What is machine learning?")
    >>>
    >>> # Batch processing
    >>> responses = model.batch_query_llm([
    ...     "What is Python?",
    ...     "Explain neural networks"
    ... ])

    References:
    ----------
    - Inference examples in https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

    """

    def __init__(
        self,
        model_name: str,
        apply_bnb_4bit_quantization: bool = True,
        system_prompt: str | None = None,
        **kwargs,
    ) -> None:
        """Initialize the LLM with specified configurations and load it onto the appropriate device.

        This constructor handles model initialization, including tokenizer setup,
        quantization configuration, and device placement. It supports both 4-bit
        quantization for memory efficiency and standard loading modes.

        Parameters
        ----------
        model_name : str
            The name or path of the model to load (e.g., "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ")
        apply_bnb_4bit_quantization : bool, optional
            Whether to apply 4-bit quantization using bitsandbytes, by default True
        system_prompt : str | None, optional
            Default system prompt to use for all interactions, by default None
        **kwargs : dict
            Additional keyword arguments for model configuration

        Raises
        ------
        Exception
            If no NVIDIA GPU is available for model execution


        Methods
        -------
        __init__(model_name: str, apply_bnb_4bit_quantization: bool = True, system_prompt: str | None = None, **kwargs) -> None
            Initialize the LLM with specified model and configurations.

        print_conversation() -> None
            Display the current conversation history in a formatted manner.

        get_model_inputs(conversation: Conversation) -> torch.Tensor
            Convert a conversation into tokenized inputs for the model.

        query_llm(prompt: str, stream: bool = True, **generation_kwargs) -> str
            Generate a response to a single prompt and update conversation history.

        batch_query_llm(prompts: list[str], **generation_kwargs) -> list[str]
            Process multiple prompts simultaneously for efficient batch inference.

        Example
        -------
        >>> # Basic initialization
        >>> model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        >>> model = LLM(model_id)
        >>>
        >>> # With custom configuration
        >>> model = LLM(
        ...     model_name=model_id,
        ...     apply_bnb_4bit_quantization=False,
        ...     system_prompt="You are a helpful AI assistant."
        ... )

        """
        if not torch.cuda.is_available():
            raise Exception("No Nvidia GPU found.")

        author, core_model_name = (
            model_name.split("/") if "/" in model_name else (None, model_name)
        )

        self.author = author
        self.model_name = model_name

        # Padding side ref: https://github.com/huggingface/transformers/issues/26061
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=False, use_fast=True, padding_side="left"
        )

        # If pad_token is None, set to EOS token
        # Refs:
        # https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954
        # https://stackoverflow.com/questions/76446228/setting-padding-token-as-eos-token-when-using-datacollatorforlanguagemodeling-fr
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if apply_bnb_4bit_quantization:
            self.default_model_configs = {
                "device_map": "auto",
                "trust_remote_code": False,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
                ),
            }
        else:
            self.default_model_configs = {
                "device_map": "auto",
                "trust_remote_code": False,
                "torch_dtype": "auto",
            }

        self.model_configs = {**self.default_model_configs, **kwargs}
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **self.model_configs)
        self.device = self.model.device
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=False)
        self.max_position_embeddings = self.config.max_position_embeddings

        # Greedy decoding with repetition penalty. Refs:
        #   https://arxiv.org/pdf/1909.05858.pdf
        #   https://huggingface.co/docs/transformers/generation_strategies
        # generate uses greedy search decoding by default so you don’t have to pass any parameters to enable it.
        # This means the parameters num_beams=1 and do_sample=False.
        self.default_generation_params = {
            "do_sample": False,  # default value
            "temperature": 1.0,  # default value
            "num_beams": 1,  # default value
            "repetition_penalty": 1.0,  # default value
            "length_penalty": 1.0,  # default value
        }

        # Initialize the chat
        self.conversation: Conversation = []
        self.system_prompt: str | None = system_prompt

        if self.system_prompt:
            self.conversation.append({"role": "system", "content": self.system_prompt})

    def set_conversation(self, conversation: Conversation) -> "LLM":
        """Set the current conversation to the provided conversation list.

        Parameters
        ----------
        conversation : Conversation
            A list of ConversationEntry objects representing the new conversation history

        Returns
        -------
        LLM
            The current LLM instance for method chaining

        Example
        -------
        >>> conversation = [
        ...     {"role": "system", "content": "You are a helpful assistant."},
        ...     {"role": "user", "content": "Hello!"}
        ... ]
        >>> llm.set_conversation(conversation)

        """
        self.conversation = conversation
        return self

    def clear_conversation(self) -> "LLM":
        """Clear the current conversation history.

        Returns:
        -------
        LLM
            The current LLM instance for method chaining

        Example:
        -------
        >>> llm.clear_conversation()

        """
        self.conversation = []
        return self

    def add_conversation_entry(self, entry: ConversationEntry) -> "LLM":
        """Add a new entry to the conversation history.

        Parameters
        ----------
        entry : ConversationEntry
            A dictionary containing 'role' and 'content' keys representing the new conversation entry

        Returns
        -------
        LLM
            The current LLM instance for method chaining

        Raises
        ------
        Exception
            If attempting to add a system entry to a non-empty conversation
        TypeError, KeyError, ValueError
            If the entry fails validation checks

        Example
        -------
        >>> entry = {"role": "user", "content": "What is machine learning?"}
        >>> llm.add_conversation_entry(entry)

        """
        validate_conversation_entry(entry)

        if entry["role"] == "system" and not len(self.conversation) == 0:
            raise Exception(
                "Trying to insert a system entry in an already populated conversation."
            )

        self.conversation.append(entry)
        return self

    def get_conversation(
        self, apply_chat_template: bool = True, add_generation_prompt: bool = False
    ) -> Conversation | str:
        """Retrieve the current conversation history.

        Parameters
        ----------
        apply_chat_template : bool, optional
            If True, applies the model's chat template to format the conversation, by default True
        add_generation_prompt : bool, optional
            If True and apply_chat_template is True, adds a generation prompt at the end, by default False

        Returns
        -------
        Conversation | str
            Either the raw conversation list or the templated conversation string, depending on apply_chat_template

        Raises
        ------
        AssertionError
            If apply_chat_template is True and the conversation is empty

        Example
        -------
        >>> # Get raw conversation
        >>> conversation = llm.get_conversation(apply_chat_template=False)
        >>>
        >>> # Get templated conversation
        >>> formatted_conversation = llm.get_conversation(apply_chat_template=True)

        """
        if apply_chat_template:
            assert self.conversation, "Current conversation is empty."

            templated_conversation = self.tokenizer.apply_chat_template(
                self.conversation, add_generation_prompt=add_generation_prompt, tokenize=False
            )

            return templated_conversation
        else:
            return self.conversation

    def get_model_inputs(self, conversation: Conversation) -> torch.Tensor:
        """Convert a conversation into tokenized inputs suitable for the model.

        This method handles the preprocessing of conversations into a format that
        can be fed directly into the language model. It applies the model's chat
        template, handles tokenization, and manages context length constraints.

        Parameters
        ----------
        conversation : Conversation
            List of conversation messages in the format:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

        Returns
        -------
        torch.Tensor
            Tokenized and formatted input tensors ready for model processing

        Raises
        ------
        Warning
            If the conversation length exceeds model's maximum position embeddings

        Example
        -------
        >>> model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        >>> model = LLM(model_id)
        >>> conversation = [
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi! How can I help?"},
        ...     {"role": "user", "content": "Tell me about Python."}
        ... ]
        >>> inputs = model.get_model_inputs(conversation)

        """
        text = self.tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(
            self.device if self.device else "cuda"
        )

        return model_inputs

    def query_llm(self, prompt: str, stream: bool = True, **generation_kwargs) -> str:
        """Generate a response to a single prompt while maintaining conversation context.

        This method handles the complete pipeline of processing a user prompt:
        adding it to the conversation history, generating a response, and updating
        the chat history with the new interaction. It supports both streaming and
        non-streaming response generation.

        Parameters
        ----------
        prompt : str
            The user's input text to generate a response for
        stream : bool, optional
            Whether to stream the output tokens in real-time, by default True
        **generation_kwargs : dict
            Additional parameters for response generation (e.g., temperature,
            max_new_tokens, top_p, etc.)

        Returns
        -------
        str
            The model's generated response text

        Example
        -------
        >>> model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        >>> model = LLM(model_id)
        >>>
        >>> # Basic query with default parameters
        >>> response = model.query_llm("What is artificial intelligence?")
        >>>
        >>> # Query with custom generation parameters
        >>> response = model.query_llm(
        ...     prompt="Write a poem about coding",
        ...     stream=False,
        ...     temperature=0.8
        ... )

        """
        _current_conversation = copy.deepcopy(self.conversation)
        _current_conversation.append({"role": "user", "content": prompt})
        model_inputs = self.get_model_inputs(_current_conversation)

        if generation_kwargs:
            generation_params = {**generation_kwargs}
        else:
            generation_params = self.default_generation_params

        streamer = (
            None
            if not stream
            else TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        )

        # Generate streamed output, visible one token at a time
        # prompt_lookup_num_tokens ref: https://twitter.com/joao_gante/status/1747322413006643259
        generated_ids = self.model.generate(
            **model_inputs,
            streamer=streamer,
            max_new_tokens=self.max_position_embeddings - model_inputs["input_ids"].shape[1],
            use_cache=True,
            **generation_params,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.conversation.append({"role": "user", "content": prompt})
        self.conversation.append({"role": "assistant", "content": response})

        del model_inputs
        del generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return response

    def batch_query_llm(self, prompts: list[str], **generation_kwargs) -> list[str]:
        """Process multiple prompts simultaneously for efficient batch inference.

        This method optimizes the processing of multiple prompts by batching them
        together, which can significantly improve throughput compared to processing
        prompts individually. It handles padding, tokenization, and generation
        for all prompts in a single forward pass.

        Parameters
        ----------
        prompts : list[str]
            List of text prompts to process
        **generation_kwargs : dict
            Additional parameters for controlling the generation process
            (e.g., temperature, top_p, num_beams)

        Returns
        -------
        list[str]
            List of generated responses corresponding to each input prompt

        Example
        -------
        >>> model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        >>> model = LLM(model_id)
        >>>
        >>> # Process multiple prompts
        >>> responses = model.batch_query_llm([
        ...     "What is deep learning?",
        ...     "Explain gradient descent",
        ...     "Define neural networks"
        ... ])
        >>>
        >>> # With custom generation parameters
        >>> responses = model.batch_query_llm(
        ...     prompts=["Write a haiku", "Tell a joke"],
        ...     temperature=0.9,
        ...     top_p=0.95
        ... )

        """
        formatted_prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=False,
            )
            for prompt in prompts
        ]

        # max_length source: https://github.com/huggingface/transformers/issues/16186
        model_inputs = self.tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_position_embeddings,
        ).to(self.device)

        # Since all sequences in input_ids are already padded to the same length,
        # this length represents the maximum length of your sequences.
        max_prompt_len = model_inputs["input_ids"].size(1)

        if generation_kwargs:
            generation_params = {**generation_kwargs}
        else:
            generation_params = self.default_generation_params

        # NOTE: prompt_lookup_num_tokens not supported for batch inference at this time
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_position_embeddings - max_prompt_len,
            **generation_params,
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        del model_inputs
        del generated_ids
        gc.collect()
        torch.cuda.empty_cache()

        return responses
