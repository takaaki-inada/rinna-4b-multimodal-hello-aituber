import argparse
import dataclasses
import io
import time
from enum import Enum, auto
from threading import Thread
from typing import Any, List, Tuple

import torch
from minigpt4.common.registry import registry
from PIL import Image
from transformers import (AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer,
                          StoppingCriteria, StoppingCriteriaList,
                          TextIteratorStreamer)


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    # system_img: List[Image.Image] = []
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        ret = ""
        for role, message in self.messages:
            if message:
                ret += role + ": " + message + self.sep
            else:
                ret += role + ": "
        print(ret)
        return ret

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            # "system_img": self.system_img,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


CONV_VISION = Conversation(
    # system="Give the following image: <Img><ImageHere></Img>. "
    #        "You will be able to see the image once I provide it to you. Please answer my questions.",
    system="<Img><ImageHere></Img>このキャラクターは何ですか？",  # 未使用
    roles=("ユーザー", "システム"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="\n",
)


class Chat:
    def __init__(self, model, vis_processor, device='cuda:0'):
        self.device = device
        self.model = model
        self.vis_processor = vis_processor
        stop_words_ids = [torch.tensor([835]).to(self.device),
                          torch.tensor([2277, 29937]).to(self.device)]  # '###' can be encoded in two different ways.
        self.stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
        self.tokenizer = self.model.gpt_neox_tokenizer
        self.streamer = TextIteratorStreamer(self.tokenizer)

    def ask(self, text, conv):
        if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
                and conv.messages[-1][1][-6:] == '</Img>':  # last message is image.
            conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
        else:
            conv.append_message(conv.roles[0], text)

    def answer(self, conv, img_list, max_new_tokens=256, num_beams=1, min_length=16, top_p=0.85,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000, is_async=False):
        conv.append_message(conv.roles[1], '')
        # embs = self.get_context_emb(conv, img_list)
        embs = self.model.get_context_emb(conv.get_prompt(), img_list)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print('Warning: The number of tokens in current conversation exceeds the max length. '
                  'The model will not see the contexts outside the range.')
        begin_idx = max(0, current_max_len - max_length)

        embs = embs[:, begin_idx:]

        generation_kwargs = dict(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        if is_async:
            generation_kwargs['streamer'] = self.streamer
            thread = Thread(target=self.model.gpt_neox_model.generate, kwargs=generation_kwargs)
            thread.start()
            return ''

        output_ids = self.model.gpt_neox_model.generate(**generation_kwargs)
        output_text = self.tokenizer.decode(output_ids.tolist()[0], skip_special_tokens=True)
        print(output_text)
        conv.messages[-1][1] = output_text
        return output_text

    def upload_img(self, image, conv, img_list):
        if isinstance(image, str):  # is a image path
            raw_image = Image.open(image).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, Image.Image):
            raw_image = image
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            if len(image.shape) == 3:
                image = image.unsqueeze(0)
            image = image.to(self.device)
        elif isinstance(image, bytes):
            raw_image = Image.open(io.BytesIO(image)).convert('RGB')
            image = self.vis_processor(raw_image).unsqueeze(0).to(self.device)

        image_emb = self.model.encode_img(image)
        img_list.append(image_emb)
        conv.append_message(conv.roles[0], "<Img><ImageHere></Img>")
        msg = "Received."
        # self.conv.append_message(self.conv.roles[1], msg)
        return msg

    def get_context_emb(self, conv, img_list):
        prompt = conv.get_prompt()
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(self.device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
