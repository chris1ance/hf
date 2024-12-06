"""Gradio chat app based on: gemma-2-9b-it: https://huggingface.co/spaces/huggingface-projects/gemma-2-9b-it/blob/main/app.py."""

from collections.abc import Iterator
from threading import Thread

import gradio as gr
import torch
from config import DEFAULT_MODEL_ID
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

model_id = DEFAULT_MODEL_ID

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_id)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16
    ),
)

model.eval()


def generate(
    message: str,
    chat_history: list[tuple[str, str]],
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

    text = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # `streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1.
    streamer = TextIteratorStreamer(
        tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True
    )

    generate_kwargs = dict(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        streamer=streamer,
        max_new_tokens=model.config.max_position_embeddings - model_inputs["input_ids"].shape[1],
        do_sample=False,
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield "".join(outputs)


chat_interface = gr.ChatInterface(
    fn=generate,
    chatbot=gr.Chatbot(height=600, show_copy_button=True),
    textbox=gr.Textbox(lines=5, stop_btn=True, submit_btn=True),
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
