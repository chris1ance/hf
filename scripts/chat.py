"""Gradio chat app based on: gemma-2-9b-it: https://huggingface.co/spaces/huggingface-projects/gemma-2-9b-it/blob/main/app.py."""

from collections.abc import Iterator
from threading import Thread

import gradio as gr
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    # attn_implementation="flash_attention_2",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    ),
)

if model_id == "google/gemma-2-9b-it":
    from transformers import GemmaTokenizerFast

    model.config.sliding_window = 4096
    tokenizer = GemmaTokenizerFast.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        # attn_implementation="flash_attention_2",   # gemma-2-9b-it currently does not work with flash_attention_2
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
    )

model.eval()

MAX_INPUT_TOKEN_LENGTH = model.config.max_position_embeddings
DEFAULT_MAX_NEW_TOKENS = round(MAX_INPUT_TOKEN_LENGTH / 4)
MAX_MAX_NEW_TOKENS = round(MAX_INPUT_TOKEN_LENGTH / 2)


def generate(
    message: str,
    chat_history: list[tuple[str, str]],
    do_sample: bool = False,
    max_new_tokens: int = 1024,
    temperature: float = 0,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1,
    num_beams: int = 1,
) -> Iterator[str]:
    """Run the interactive chat application."""
    conversation = []
    for user, assistant in chat_history:
        conversation.extend(
            [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        )
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt"
    )
    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]
        gr.Warning(
            f"Trimmed input from conversation as it was longer than {MAX_INPUT_TOKEN_LENGTH} tokens."
        )
    input_ids = input_ids.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer, timeout=30.0, skip_prompt=True, skip_special_tokens=True
    )
    generate_kwargs = dict(
        {"input_ids": input_ids},
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
        num_beams=num_beams,
        repetition_penalty=repetition_penalty,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


chat_interface = gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(height=600),
    textbox=gr.Textbox(lines=2),
    additional_inputs=[
        gr.Checkbox(label="do_sample", value=False),
        gr.Slider(
            label="Max new tokens",
            minimum=1,
            maximum=MAX_MAX_NEW_TOKENS,
            step=1,
            value=DEFAULT_MAX_NEW_TOKENS,
        ),
        gr.Slider(
            label="Temperature",
            minimum=0.1,
            maximum=4.0,
            step=0.1,
            value=0,
        ),
        gr.Slider(
            label="Top-p (nucleus sampling)",
            minimum=0.05,
            maximum=1.0,
            step=0.05,
            value=0.9,
        ),
        gr.Slider(
            label="Top-k",
            minimum=1,
            maximum=1000,
            step=1,
            value=50,
        ),
        gr.Slider(
            label="Repetition penalty",
            minimum=1.0,
            maximum=2.0,
            step=0.05,
            value=1,
        ),
        gr.Slider(
            label="num_beams",
            minimum=1,
            maximum=10,
            step=1,
            value=1,
        ),
    ],
    stop_btn=gr.Button("Stop"),
    examples=[
        ["Hello there! How are you doing?"],
        ["Can you explain briefly to me what is the Python programming language?"],
        ["Explain the plot of Cinderella in a sentence."],
        ["How many hours does it take a man to eat a Helicopter?"],
        ["Write a 100-word article on 'Benefits of Open-Source in AI research'"],
    ],
)

if __name__ == "__main__":
    chat_interface.launch()
