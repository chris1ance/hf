"""Pytest unit tests for the LLM class using an actual, small model and tokenizer.

Note:
- These tests use a real model and tokenizer from Hugging Face (a tiny test model) to ensure actual inference.
- They require an internet connection to download the model and tokenizer.
- The provided model 'sshleifer/tiny-gpt2' is a small model suitable for tests and can run on CPU.
- Make sure that the original LLM code does not strictly require CUDA. If it does, consider modifying the LLM
  class to allow CPU inference or ensure you have a CUDA-capable device.

Run the tests using:
    pytest -s test_llm_no_mock.py

"""

import pytest
import torch

# Import the LLM class from your module
from hf.llm import LLM  # Replace with the actual module name


@pytest.fixture(scope="session")
def model_name():
    """Name of huggingface model to load."""
    return "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="session")
def llm_instance(model_name):
    """Initialize LLM without quantization for simplicity."""
    model = LLM(
        model_name=model_name, apply_bnb_4bit_quantization=False, system_prompt="System prompt"
    )
    return model


def test_init(llm_instance):
    """Test that the LLM initializes correctly with an actual model."""
    assert llm_instance.model_name is not None
    assert llm_instance.system_prompt == "System prompt"
    assert len(llm_instance.conversation) == 1
    assert llm_instance.conversation[0]["role"] == "system"
    assert llm_instance.conversation[0]["content"] == "System prompt"


def test_set_conversation(llm_instance):
    """Test that set_conversation updates the conversation."""
    new_conversation = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    llm_instance.set_conversation(new_conversation)
    assert llm_instance.conversation == new_conversation


def test_clear_conversation(llm_instance):
    """Test that clear_conversation clears the conversation."""
    llm_instance.clear_conversation()
    assert len(llm_instance.conversation) == 0


def test_get_conversation_with_template(llm_instance):
    """Test get_conversation applies template when requested."""
    # Add a conversation so it's not empty
    llm_instance.set_conversation([{"role": "user", "content": "Hello"}])

    conv_output = llm_instance.get_conversation(apply_chat_template=False)
    assert isinstance(conv_output, list)

    conv_output_template = llm_instance.get_conversation(apply_chat_template=True)
    assert isinstance(conv_output_template, str)

    conv_output_template_gen = llm_instance.get_conversation(
        apply_chat_template=True, add_generation_prompt=True
    )
    assert isinstance(conv_output_template_gen, str)
    assert len(conv_output_template_gen) > len(conv_output_template)


def test_get_model_inputs(llm_instance):
    """Test that get_model_inputs returns a torch.Tensor with actual tokenizer output."""
    conversation = [
        {"role": "user", "content": "What's up?"},
        {"role": "assistant", "content": "All good!"},
    ]
    inputs = llm_instance.get_model_inputs(conversation)
    assert "input_ids" in inputs and "attention_mask" in inputs
    assert isinstance(inputs["input_ids"], torch.Tensor)
    assert isinstance(inputs["attention_mask"], torch.Tensor)


def test_query_llm(llm_instance):
    """Test query_llm generates a response and updates conversation using real inference."""
    response = llm_instance.query_llm("Hello?", stream=False)
    assert isinstance(response, str)
    # Conversation should be appended
    assert llm_instance.conversation[-2]["role"] == "user"
    assert llm_instance.conversation[-2]["content"] == "Hello?"
    assert llm_instance.conversation[-1]["role"] == "assistant"
    assert isinstance(llm_instance.conversation[-1]["content"], str)


def test_batch_query_llm(llm_instance):
    """Test batch_query_llm generates responses for multiple prompts using real inference."""
    prompts = ["Question 1", "Question 2"]
    responses = llm_instance.batch_query_llm(prompts)
    assert len(responses) == 2
    for resp in responses:
        assert isinstance(resp, str)
