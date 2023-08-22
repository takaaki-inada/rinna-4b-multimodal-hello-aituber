import asyncio
import os
import random
import re
import sys

import gradio as gr
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aituber.conversation import CONV_VISION, Chat
from aituber.customized_mini_gpt4 import CustomizedMiniGPT4
# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor
from minigpt4.runners import *
from minigpt4.tasks import *

is_use_expression = False
if is_use_expression:
    from aituber.expression import get_expression


# ========================================
#             Model Initialization
# ========================================
print("Initializing Chat")
model = CustomizedMiniGPT4(gpt_neox_model="rinna/bilingual-gpt-neox-4b", low_resource=True)
tokenizer = model.gpt_neox_tokenizer
if torch.cuda.is_available():
    model = model.to("cuda")

ckpt_path = "./checkpoint.pth"
print("Load BLIP2-LLM Checkpoint: {}".format(ckpt_path))
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt['model'], strict=False)

vis_processor = Blip2ImageEvalProcessor()

chat = Chat(model, vis_processor)
print("Initialization Finished")


# ========================================
#             Gradio Setting
# ========================================
def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return (
        None,
        gr.update(value=None, interactive=True),
        gr.update(placeholder="Please upload your image first", interactive=False),
        gr.update(value="Upload & Start Chat", interactive=True),
        chat_state,
        img_list,
    )


def upload_img(gr_img, text_input, chat_state):
    if gr_img is None:
        return None, None, gr.update(interactive=True), chat_state, None
    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return (
        gr.update(interactive=False),
        gr.update(interactive=True, placeholder="Type and press Enter"),
        gr.update(value="Start Chatting", interactive=False),
        chat_state,
        img_list,
    )


api_chat_history = CONV_VISION.copy()
api_img_list = []


def sanitize_output(output, chatbot):
    output = output.replace("<s>", "")
    output = output.replace("</s>", "")
    output = output.replace("システム: ", "")
    output = re.sub(r'\<.*?\>', '', output)
    output = re.sub(r'\[.*?\]', '', output)
    if len(chatbot[-1][1]) == 0:
        output = output.replace(" ", "")
    if not output:
        return output
    output = output.replace("<NL>", "\n")
    output = output.replace("_", " ")
    if len(chatbot[-1][1]) != 0:
        if re.search(r'[ぁ-ん]+|[ァ-ヴー]+', output):
            # NOTE: なぜか日本語の場合に最後の文字と最初の文字が重複する場合があるため、簡易的に最後の文字を削除
            output = output.strip()
            if chatbot[-1][1][-1] == output[0]:
                output = output[1:]
        elif chatbot[-1][1][-1] != " " and output != " ":
            # NOTE: 英語が返ってくる場合streamが1word毎に返ってくるため、簡易的にword間に半角スペースを1文字入れてみた
            output = " " + output
    return output


def reset_api():
    global api_chat_history, api_img_list
    api_chat_history = CONV_VISION.copy()
    api_img_list = []


def upload_image_from_api(image: bytes):
    chat.upload_img(image, api_chat_history, api_img_list)


async def generate_from_api(texts, max_new_tokens=256, min_length=16, temperature=1.0, post_process=None):
    old_expression = None
    for i, ts in enumerate(texts):
        if isinstance(ts, str):
            text = ts
        elif isinstance(ts, list):
            text = ts[random.randint(0, len(ts) - 1)]
        else:
            raise NotImplementedError
        chat.ask(text, api_chat_history)
        llm_message = chat.answer(
            conv=api_chat_history,
            img_list=api_img_list,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            num_beams=1,
            temperature=temperature,
            max_length=2000,
            is_async=True,
        )
        api_chat_history.messages[-1][1] = llm_message
        for output in chat.streamer:
            output = sanitize_output(output, api_chat_history.messages)
            if not output:
                continue
            if post_process:
                output = post_process(text, output, api_chat_history.messages)
            # NOTE: historyにはexpressionを含めずに記録
            api_chat_history.messages[-1][1] += output
            if is_use_expression:
                expression = get_expression(output)
                if expression and expression != old_expression:
                    output = f'[{expression}]{output}'
                    old_expression = expression
            yield output


def gradio_ask(user_message, chatbot, chat_state):
    if len(user_message) == 0:
        return (
            gr.update(interactive=True, placeholder="Input should not be empty!"),
            chatbot,
            chat_state,
        )
    chat.ask(user_message, chat_state)
    chatbot = chatbot + [[user_message, None]]
    return "", chatbot, chat_state


def gradio_answer(chatbot, chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(
        conv=chat_state,
        img_list=img_list,
        max_new_tokens=256,
        min_length=16,
        num_beams=1,
        temperature=temperature,
        max_length=2000,
        is_async=True,
    )
    chatbot[-1][1] = llm_message
    return chatbot, chat_state, img_list


async def bot(chatbot, chat_state):
    old_expression = None
    for output in chat.streamer:
        await asyncio.sleep(0)
        output = sanitize_output(output, chatbot)
        if not output:
            continue
        if is_use_expression:
            expression = get_expression(output)
            if expression and expression != old_expression:
                output = f'[{expression}]{output}'
                old_expression = expression
        print(output)
        chatbot[-1][1] += output
        chat_state.messages[-1][1] += output
        yield chatbot, chat_state


title = """<h1 align="center">MiniGPT-4(rinna/bilingual-gpt-neox-4b)</h1>"""
description = (
    """<h3>This is webui of MiniGPT-4 by combining GPT-NeoX rinna/bilingual-gpt-neox-4b model and BLIP-2. Upload your images and start chatting!</h3>"""
)
article = """
"""

default_img = "https://huggingface.co/rinna/bilingual-gpt-neox-4b-minigpt4/resolve/main/sample.jpg"

with gr.Blocks() as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown(article)

    with gr.Row():
        with gr.Column(scale=0.5):
            image = gr.Image(type="pil", value=default_img)
            upload_button = gr.Button(
                value="Upload & Start Chat", interactive=True, variant="primary"
            )
            clear = gr.Button("Restart")

            num_beams = gr.Slider(
                minimum=1,
                maximum=5,
                value=1,
                step=1,
                interactive=True,
                label="beam search numbers)",
            )

            temperature = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                interactive=True,
                label="Temperature",
            )

        with gr.Column():
            chat_state = gr.State()
            img_list = gr.State()
            chatbot = gr.Chatbot(label="MiniGPT-4")
            text_input = gr.Textbox(
                label="User",
                placeholder="Please upload your image first",
                interactive=False,
            )

    upload_button.click(
        upload_img,
        [image, text_input, chat_state],
        [image, text_input, upload_button, chat_state, img_list],
    )

    response = text_input.submit(
        gradio_ask, [text_input, chatbot, chat_state], [text_input, chatbot, chat_state], queue=False
    ).then(
        gradio_answer,
        [chatbot, chat_state, img_list, num_beams, temperature],
        [chatbot, chat_state, img_list],
        queue=False
    ).then(
        bot, [chatbot, chat_state], [chatbot, chat_state],
    )
    response.then(lambda: gr.update(interactive=True), None, [text_input], queue=False)
    clear.click(
        gradio_reset,
        [chat_state, img_list],
        [chatbot, image, text_input, upload_button, chat_state, img_list],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch(debug=True, enable_queue=True)
