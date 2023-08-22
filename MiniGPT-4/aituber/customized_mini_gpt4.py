import torch
import torch.nn as nn
from minigpt4.models.blip2 import Blip2Base, disabled_train
from minigpt4.models.mini_gpt4 import MiniGPT4
from peft import PeftConfig, PeftModel
from transformers import AutoTokenizer
from transformers.models.gpt_neox import GPTNeoXForCausalLM


class CustomizedGPTNeoXForCausalLM(GPTNeoXForCausalLM):
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs):
        input_shape = input_ids.shape

        # cut decoder_input_ids if past is used
        if past_key_values and past_key_values[0] is not None:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
            }
        )
        return model_inputs


class CustomizedMiniGPT4(Blip2Base):
    """
    BLIP2 GPT-NeoX model.
    """
    def __init__(
        self,
        gpt_neox_model="rinna/bilingual-gpt-neox-4b",
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
        lora_model=None,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT', flush=True)
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            print("freeze vision encoder")
        print('Loading VIT Done')

        print('Loading Q-Former', flush=True)
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            print("freeze Qformer")
        print('Loading Q-Former Done')

        print('Loading LLM', flush=True)
        self.gpt_neox_tokenizer = AutoTokenizer.from_pretrained(gpt_neox_model, use_fast=False)

        if self.low_resource:
            self.gpt_neox_model_base = CustomizedGPTNeoXForCausalLM.from_pretrained(
                gpt_neox_model,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map={'': device_8bit}
            )
        else:
            self.gpt_neox_model_base = CustomizedGPTNeoXForCausalLM.from_pretrained(
                gpt_neox_model,
                torch_dtype=torch.float16,
            )

        for name, param in self.gpt_neox_model_base.named_parameters():
            param.requires_grad = False
        print('Loading LLM Done')

        if lora_model:
            print('Loading LoRA')
            # LoRAモデルの準備
            self.gpt_neox_model = PeftModel.from_pretrained(
                self.gpt_neox_model_base,
                lora_model,
                # device_map="auto"
            )
            print('Loading LoRA Done')
        else:
            self.gpt_neox_model = self.gpt_neox_model_base

        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.gpt_neox_model.config.hidden_size
        )

    def vit_to_cpu(self):
        MiniGPT4.vit_to_cpu(self)

    def encode_img(self, image):
        inputs_gpt_neox, _ = MiniGPT4.encode_img(self, image)
        return inputs_gpt_neox

    def get_context_emb(self, prompt, img_list):
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            self.gpt_neox_tokenizer(seg, return_tensors="pt", add_special_tokens=False).to(self.device).input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.gpt_neox_model.gpt_neox.embed_in(seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        return mixed_embs
