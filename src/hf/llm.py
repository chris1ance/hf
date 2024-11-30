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

# Type alias for a single chat message tuple (user message, assistant message)
ChatMessage = tuple[str, str]

# Type alias for the chat history, which is a list of chat messages
ChatHistory = list[ChatMessage]


# Type alias for a conversation entry (role: "user" or "assistant", content: message content)
class ConversationEntry(TypedDict):
    """Type alias for a conversation entry (role: "user" or "assistant", content: message content)."""

    role: str
    content: str


# Type alias for a conversation, which is a list of conversation entries
Conversation = list[ConversationEntry]

########################################################################################


class BaseLLM:
    """Base class for a language model (LLM) that manages conversation history and system prompts.

    Attributes
    ----------
        system_prompt (str | None): Optional initial prompt used to set the context for the conversation.
        chat_history (list[tuple[str, str]]): List to store the chat history, where each entry is a tuple containing user and assistant messages.

    Methods
    -------
        set_system_prompt: Sets the system prompt for the conversation.
        clear_chat_history: Clears the current chat history.
        get_chat_history: Retrieves the current chat history.
        set_chat_history: Sets the chat history to a specified list of message tuples.
        get_conversation: Converts the chat history to a structured conversation format.

    """

    def set_system_prompt(self, system_prompt: str | None) -> "BaseLLM":
        """Set the system prompt for the conversation.

        Args:
        ----
            system_prompt (str | None): The system prompt to be set. If None, no system prompt is set.

        Returns:
        -------
            BaseLLM: An instance of the class with the updated system prompt.

        """
        if system_prompt:
            self.system_prompt = system_prompt
        return self

    def clear_chat_history(self) -> "BaseLLM":
        """Clear the chat history.

        This method resets the chat history, removing all stored conversation tuples.

        Returns
        -------
            BaseLLM: An instance of the class with the chat history cleared.

        """
        self.chat_history = []
        return self

    def get_chat_history(self) -> ChatHistory:
        """Get the current chat history.

        Retrieves the entire chat history as a list of tuples, where each tuple represents a message exchange.

        Returns
        -------
            ChatHistory: A list of ChatMessages representing the chat history. Each ChatMessage contains user and assistant messages.

        """
        return self.chat_history

    def set_chat_history(self, chat_history: ChatHistory) -> "BaseLLM":
        """Set the chat history to a specific state.

        Args:
        ----
            chat_history: A ChatHistory representing the chat history to be set.

        Returns:
        -------
            BaseLLM: An instance of the class with the updated chat history.

        """
        self.chat_history = chat_history
        return self

    def get_conversation(self) -> Conversation:
        """Convert the chat history into a structured conversation format.

        Returns
        -------
            Conversation: A list of ConversationEntries representing the structured conversation.

        """
        conversation = []

        if self.system_prompt:
            conversation.append({"role": "system", "content": self.system_prompt})
        for user, assistant in self.chat_history:
            conversation.extend(
                [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
            )

        return conversation


########################################################################################


class LLM(BaseLLM):
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
        self.chat_history: ChatHistory = []
        self.system_prompt: str | None = system_prompt

    def print_conversation(self) -> None:
        """Display the current conversation history in a formatted, human-readable manner.

        This method prints the entire conversation history, including both user inputs
        and model responses, using the model's chat template for consistent formatting.
        It's useful for debugging and reviewing the conversation flow.

        Returns:
        -------
        None

        Example:
        -------
        >>> model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        >>> model = LLM(model_id)
        >>> model.query_llm("Hello!")
        >>> model.query_llm("How are you?")
        >>> model.print_conversation()

        """
        conversation = self.get_conversation()
        print(
            self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=False, tokenize=False
            )
        )
        return None

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
        ...     temperature=0.8,
        ...     max_new_tokens=100
        ... )

        """
        conversation = self.get_conversation()
        conversation.append({"role": "user", "content": prompt})
        model_inputs = self.get_model_inputs(conversation)

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

        self.chat_history.append((prompt, response))

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
