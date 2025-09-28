# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import warnings
from typing import Any, List, Optional, Tuple, Union

import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from transformers import AutoModelForSpeechSeq2Seq

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel
from .audio import AudioEncoder, process_audio, process_audiov2, Qwen2AudioMultiModalProjector
from .modeling_intern_vit_adapter import InternVisionModelAdapter
from detectron2.modeling import build_sem_seg_head, ShapeSpec
from .mask_decoder import setup, loss_labels, loss_masks
from internvl.model.whisper.modeling_whisper import AudioWhisperModel

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    _no_split_modules = ['InternVisionModel', 'LlamaDecoderLayer', 'InternLM2DecoderLayer',
                         'Phi3DecoderLayer', 'Qwen2DecoderLayer']
    _supports_flash_attn_2 = True

    def __init__(self, config: InternVLChatConfig, vision_model=None, language_model=None, audio_model=None):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        
        self.enable_mask = config.enable_mask

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            if not self.enable_mask:
                self.vision_model = InternVisionModel(config.vision_config)
            else:
                self.vision_model = InternVisionModelAdapter(config.vision_config)
                checkpoint = torch.load("playground/pretrained/model_final.pth", map_location='cpu')['model']
                filtered_state_dict = {k.replace('backbone.', ''): v for k, v in checkpoint.items() if k.startswith('backbone')}
                self.vision_model.load_state_dict(filtered_state_dict, strict=True)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')
            
        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )

        # audio ============================================================================
        self.config.qwa_version = getattr(self.config, 'qwa_version', "internomni")
        if audio_model is not None:
            self.audio_model = audio_model
        elif getattr(self.config, 'qwa_version', "internomni") == "v1":
            audio_config = {
                "n_mels": 80,
                "n_ctx": 1500,
                "n_state": 1280,
                "n_head": 20,
                "n_layer": 32,
                "output_dim": 4096,
                "avg_pool": True,
                "add_audio_bos_eos_token": True,
                "audio_start_id": 155163
            }
            self.audio_model = AudioEncoder(**audio_config)
            # if config.audio_encoder_path:
            self.audio_model.load_state_dict(torch.load("pretrained/QWen-Audio-Chat/audio_encoder_weights.pth", map_location='cpu'), strict=False)
            self.mlp2 = nn.Sequential(
                nn.LayerNorm(4096),
                nn.Linear(4096, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
            # mlp2_weight = torch.load("pretrained/InternVL2-1B/mlp_weight.pth", map_location="cpu")
            # self.mlp2.load_state_dict(mlp2_weight, strict=False)
        elif getattr(self.config, 'qwa_version', "internomni") == "v2":
            self.audio_model = AutoModelForSpeechSeq2Seq.from_pretrained("pretrained/whisper-large-v3").model
            del self.audio_model.decoder
            
            self.mlp2 = nn.Sequential(
                nn.LayerNorm(1280),
                nn.Linear(1280, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size))
        elif getattr(self.config, 'qwa_version', "internomni") == "internomni":
            self.audio_model = AudioWhisperModel.from_pretrained("playground/pretrained/internomni_whisper")
            
            self.mlp2 = nn.Sequential(
                nn.LayerNorm(1280),
                nn.Linear(1280, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )
            # load whisper-large-v3-turbo
            if llm_hidden_size == 4096:
                self.mlp2.load_state_dict(torch.load('pretrained/internomni_whisper/mlp_projector.pth', map_location='cpu'), strict=True, assign=True)
        elif getattr(self.config, 'qwa_version', "internomni") == "turbo":
            self.audio_model = AutoModelForSpeechSeq2Seq.from_pretrained("pretrained/whisper-large-v3-turbo").model
            del self.audio_model.decoder
            self.mlp2 = nn.Sequential(
                nn.LayerNorm(1280),
                nn.Linear(1280, llm_hidden_size),
                nn.GELU(),
                nn.Linear(llm_hidden_size, llm_hidden_size)
            )            
        else:
            raise NotImplementedError
        # audio ============================================================================
        if self.enable_mask:
            cfg = setup()
            cfg.defrost()
            cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
            cfg.MODEL.SEM_SEG_HEAD.SEG_TOKEN_DIM = llm_hidden_size
            # 是否使用 ps
            # cfg.MODEL.SEM_SEG_HEAD.USE_PS = True
            cfg.MODEL.SEM_SEG_HEAD.USE_PS = False
            
            # cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2
            stride = {"res2": 4, "res3": 8, "res4": 16, "res5": 32}
            input_channel = {
                "res2": 1024,
                "res3": 1024,
                "res4": 1024,
                "res5": 1024,
            }

            shape_list = {
                name: ShapeSpec(channels=input_channel[name], stride=stride[name]) for name in ["res2", "res3", "res4", "res5"]
            }
            cfg.freeze()
            self.mask_decoder = build_sem_seg_head(cfg, shape_list)
            
        else:
            print("Mask decoder is not used")

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora)

        if config.use_llm_lora:
            self.wrap_llm_lora(r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora)
        
        if config.use_audio_lora:
            self.wrap_audio_lora(r=config.use_audio_lora, lora_alpha=2 * config.use_audio_lora)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model.encoder = get_peft_model(self.vision_model.encoder, lora_config)
        # self.vision_model.embeddings = get_peft_model(self.vision_model.embeddings, lora_config)
        # self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                              'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM'
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()
    
    def wrap_audio_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_alpha = 2 * lora_alpha
        if self.config.qwa_version == "v1":
            lora_config = LoraConfig(
                r=r,
                target_modules=['attn.query', 'attn.key', 'attn.value', 'attn.out', 'mlp.0', 'mlp.2'],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.audio_model = get_peft_model(self.audio_model, lora_config)
            self.audio_model.print_trainable_parameters()
        elif self.config.qwa_version == "v2":
            lora_config = LoraConfig(
                r=r,
                target_modules=['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj', 'attn.out', 'fc1', 'fc2', 'self_attn.out_proj'],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.audio_model = get_peft_model(self.audio_model, lora_config)
            self.audio_model.print_trainable_parameters()
        else:
            lora_config = LoraConfig(
                r=r,
                target_modules=['self_attn.k_proj', 'self_attn.q_proj', 'self_attn.v_proj', 'attn.out', 'fc1', 'fc2', 'self_attn.out_proj'],
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            self.audio_model = get_peft_model(self.audio_model, lora_config)
            self.audio_model.print_trainable_parameters()

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        input_audios = None,
        input_audio_lengths = None,
        audio_flags = None,
        image_num = None,
        masks = None,
        mask_flag=None,
        dense_frame_flag=None,
        seg_image_num=None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds, ms_feat = self.extract_feature(pixel_values)
        # if dense_frame_flag is not None:
        #     vit_embeds = vit_embeds[dense_frame_flag == 1]
        if dense_frame_flag is not None and torch.any(dense_frame_flag == 0):
            # Pool tokens for frames where dense_frame_flag is 0
            vit_embeds = vit_embeds[image_flags == 1]
            dense_frame_flag = dense_frame_flag[image_flags == 1]
            
            T, num_tokens, C = vit_embeds.shape
            pooled_embeds = []
            for i in range(T):
                if dense_frame_flag[i] == 0:
                    # Average pool the 256 tokens into 1 token
                    pooled_token = vit_embeds[i].mean(dim=0, keepdim=True) # [1, C]
                    pooled_embeds.append(pooled_token)
                else:
                    # Keep all 256 tokens
                    pooled_embeds.append(vit_embeds[i])
            resample_vit_embeds = torch.cat(pooled_embeds, dim=0)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + resample_vit_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                resample_vit_embeds = resample_vit_embeds.reshape(-1, C)
                print(f'++++++++++++warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                    f'vit_embeds.shape={resample_vit_embeds.shape}')
                print(f'image_flags: {image_flags}, dense_frame_flag: {dense_frame_flag}')
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + resample_vit_embeds[:n_token]
                ignore_flag = True
            
        else:
            vit_embeds = vit_embeds[image_flags == 1]
            vit_batch_size = pixel_values.shape[0]
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')
            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            try:
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                vit_embeds = vit_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                    f'vit_embeds.shape={vit_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
                ignore_flag = True
        
        input_embeds = input_embeds.reshape(B, N, C)
        input_ids = input_ids.reshape(B, N)

        if input_audios is not None:
            if self.config.qwa_version == "v1":
                extract_audio_features = self.extract_audio_features_v1
            elif self.config.qwa_version == "v2":
                extract_audio_features = self.extract_audio_features_v2
            else:
                extract_audio_features = self.extract_audio_features_v2

            audio_select_mask = audio_flags > 0
            audio_embeds_list = extract_audio_features(input_audios, input_audio_lengths)
            
            # Filter out audio embeddings where mask is False
            filtered_audio_embeds = [emb for emb, mask in zip(audio_embeds_list, audio_select_mask) if mask]
            if len(filtered_audio_embeds) > 0:
                selected = (input_ids == self.aud_context_token_id).reshape(-1)
                input_embeds = input_embeds.reshape(-1, C)
                audio_embeds = torch.cat(filtered_audio_embeds, dim=0)
                input_embeds[selected] = input_embeds[selected] * 0.0 + audio_embeds
            
            sub_audio_select_mask = audio_flags == 2
            filtered_sub_audio_embeds = [emb for emb, mask in zip(audio_embeds_list, sub_audio_select_mask) if mask]
            if len(filtered_sub_audio_embeds) > 0:
                sub_audio_embeds = torch.cat(filtered_sub_audio_embeds, dim=0)
                sub_audio_selected = (input_ids == self.sub_aud_context_token_id).reshape(-1)
                input_embeds[sub_audio_selected] = input_embeds[sub_audio_selected] * 0.0 + sub_audio_embeds
            
            input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0
        
        if self.enable_mask:
            if dense_frame_flag is not None:
                seg_image_flag = torch.ones_like(dense_frame_flag == 1)
                if seg_image_flag.any():
                    ms_feat = [feat[seg_image_flag] for feat in ms_feat]
                    image_num = seg_image_num
                else:
                    pass
                
            loss_cls, loss_mask, loss_dice = self.mask_loss(ms_feat, input_ids, outputs.hidden_states[-1], image_num, masks, image_flags, mask_flag)
            # Save individual losses to tensorboard
            # if self.config.use_tensorboard:
            import torch.distributed as dist
            if dist.get_rank() == 0:
                if not hasattr(self, 'writer'):
                    from torch.utils.tensorboard import SummaryWriter
                    import datetime
                    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                    log_dir = f"log_dir/{current_time}"
                    self.writer = SummaryWriter(log_dir=log_dir)
                    self.global_step = 0
                self.writer.add_scalar('Loss/llm', loss.float(), self.global_step)
                self.writer.add_scalar('Loss/cls', loss_cls.float(), self.global_step)
                self.writer.add_scalar('Loss/mask', loss_mask.float(), self.global_step)
                self.writer.add_scalar('Loss/dice', loss_dice.float(), self.global_step)
            
            loss = loss + 1.0 * loss_cls + 2.0 * loss_mask + 0.5 * loss_dice
            
            if torch.isnan(loss) or loss == 0.0:
                print("Loss is zero or NaN.")
                import sys
                sys.exit(1)
            
            if dist.get_rank() == 0:    
                self.writer.add_scalar('Loss/total', loss.float(), self.global_step)
                self.global_step += 1

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def mask_loss(self, ms_feat, input_ids, hidden_states, image_num, masks, image_flags, mask_flag):
        seg_token_mask = input_ids == self.seg_token_id
        masks = masks[mask_flag.bool()]

        if not seg_token_mask.any():
            seg_hidden_states = hidden_states[0, :1]
            
            padded_ms_feats_list = [feat[:1].unsqueeze(1) for feat in ms_feat]
            input_feats = {k: v for k, v in zip(["res2", "res3", "res4", "res5"], padded_ms_feats_list)}
            padded_mask = torch.ones((1, 1), dtype=torch.bool, device=self.device)

            out = self.mask_decoder(input_feats, seg_hidden_states, padded_mask)

            loss_cls = torch.sum(out['pred_logits']) * 0.
            loss_dice = torch.sum(out['pred_masks']) * 0.
            loss_mask = torch.sum(out['pred_masks']) * 0.
            
            return loss_cls, loss_dice, loss_mask
        
        # assert ms_feat[0].shape[0] == torch.sum(image_num), "Wrong: Number of image!!!"

        seg_hidden_states = hidden_states[seg_token_mask]  # (N_seg, C)
        seg_token_counts = seg_token_mask.sum(dim=1)  # (N_seq, )

        cumulative_image_nums = torch.cat([torch.tensor([0]).to(seg_token_counts.device), image_num.cumsum(dim=0)]).to(seg_token_counts.device)  # (N_seq+1, )

        total_seg_tokens = seg_token_counts.sum()

        max_num_images = image_num.max()
        token_indices = torch.repeat_interleave(cumulative_image_nums[:-1], seg_token_counts)
        
        padding_lengths = image_num.repeat_interleave(seg_token_counts)
        padded_ms_feats_list = list()
        
        for ms_feat_ in ms_feat:
            padded_ms_feats = torch.zeros((total_seg_tokens, max_num_images, *ms_feat_.shape[-3:]), dtype=ms_feat_.dtype, device=self.device)
            for i in range(total_seg_tokens):
                padded_ms_feats[i, :padding_lengths[i]] = ms_feat_[token_indices[i]:token_indices[i] + padding_lengths[i]]
            padded_ms_feats_list.append(padded_ms_feats)

        padded_mask = torch.zeros((total_seg_tokens, max_num_images), dtype=torch.bool, device=self.device)
        mask_list = []
        cumulative_mask_nums = padding_lengths.cumsum(dim=0) - padding_lengths.cumsum(dim=0)[0]
        padded_gt_masks = torch.zeros((total_seg_tokens, max_num_images, *masks.shape[-2:]), dtype=torch.bool, device=self.device)
        for i in range(total_seg_tokens):
            _mask = masks[cumulative_mask_nums[i]: cumulative_mask_nums[i] + padding_lengths[i]]
            mask_list.append(_mask)
            padded_gt_masks[i, :padding_lengths[i]] = _mask
            padded_mask[i, :padding_lengths[i]] = 1
        
        masks = torch.cat(mask_list, dim=0)
        
        input_feats = {k: v for k, v in zip(["res2", "res3", "res4", "res5"], padded_ms_feats_list)}

        out = self.mask_decoder(input_feats, seg_hidden_states, padded_mask)

        pred_logits = out['pred_logits']  # (4, 10)
        pred_masks = out['pred_masks']  # (4, 10, 112, 112)

        target_label = (padded_mask & torch.any(padded_gt_masks > 0, dim=(2, 3)))[padded_mask].flatten()

        target_label = target_label.logical_not().long()
        pred_logits = pred_logits.flatten(0, 1)[padded_mask.flatten()]

        loss_cls = loss_labels(pred_logits, target_label)
        pred_masks = pred_masks[padded_mask]

        # num_masks = sum(len(t["labels"]) for t in targets)
        # num_masks = torch.as_tensor(
        #     [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        # )
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_masks)
        # num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        loss_mask, loss_dice = loss_masks(pred_masks, masks, torch.sum(1 - target_label))

        return loss_cls, loss_dice, loss_mask

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            if self.enable_mask:
                vit_embeds, ms_feat = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True)
            else:
                vit_embeds = self.vision_model(
                    pixel_values=pixel_values,
                    output_hidden_states=False,
                    return_dict=True)
                ms_feat = None
            vit_embeds = vit_embeds.last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)

        return vit_embeds, ms_feat
    
    def extract_audio_features_v1(self, input_audios, input_audio_lengths):
        audio_embeds = self.audio_model.encode(input_audios, input_audio_lengths)
        audio_embeds = self.mlp2(audio_embeds)

        output_audios = []
        for i in range(input_audios.shape[0]):
            audio_span = input_audio_lengths[i][1]
            audio = audio_embeds[i][:audio_span]
            # if bos is not None:
            #     audio = torch.concat([bos, audio, eos])
            assert len(audio) == audio_span.item()
            output_audios.append(audio)

        return output_audios
    
    def extract_audio_features_v2(self, input_audios, input_audio_lengths):
        # audio attention map
        # batch_size, _, max_mel_seq_len = input_audios.shape

        # audio_feat_lengths = input_audio_lengths[:, 0]
        # audio_output_lengths = input_audio_lengths[:, 1]

        # max_seq_len = (max_mel_seq_len - 2) // 2 + 1
        # seq_range = (
        #     torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
        #     .unsqueeze(0)
        #     .expand(batch_size, max_seq_len))
        
        # lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
        # # Create mask
        # padding_mask = seq_range >= lengths_expand

        # audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
        #     batch_size, 1, max_seq_len, max_seq_len
        # )
        # audio_attention_mask = audio_attention_mask_.to(
        #     dtype=self.audio_model.conv1.weight.dtype, device=self.audio_model.conv1.weight.device
        # )
        # audio_attention_mask[audio_attention_mask_] = float("-inf")
        
        audio_values = input_audios
        audio_len_after_cnn = input_audio_lengths[:, 0]
        
        #TODO: construct audio padding_mask in loader
        max_len_in_batch = int(torch.max(audio_len_after_cnn).item())

        padding_mask = torch.ones([audio_values.size(0), max_len_in_batch]).to(dtype=audio_values.dtype, device=audio_values.device)
        for index in range(len(audio_values)):
            padding_mask[index, :int(audio_len_after_cnn[index].item())] = 0

        last_hidden_state = self.audio_model(audio_values, padding_mask, audio_len_after_cnn)  # (bs, max_token_num, 1280)

        audio_embeds = self.mlp2(last_hidden_state)

        # #TODO: construct audio padding_mask in loader
        # max_len_in_batch = int(torch.max(audio_len_after_cnn).item())

        # padding_mask = torch.ones([audio_values.size(0), max_len_in_batch]).to(dtype=audio_values.dtype,
        #                                                                        device=audio_values.device)
        # for index in range(len(audio_values)):
        #     padding_mask[index, :int(audio_len_after_cnn[index].item())] = 0

        # last_hidden_state = self.audio_model.encoder(audio_values, padding_mask).last_hidden_state  # (bs, max_token_num, 1280)

        # audio_embeds = self.mlp2(last_hidden_state)
        output_audios = []
        for i in range(input_audios.shape[0]):
            audio_span = input_audio_lengths[i][1]
            audio = audio_embeds[i][:audio_span]
            # if bos is not None:
            #     audio = torch.concat([bos, audio, eos])
            assert len(audio) == audio_span.item()
            output_audios.append(audio)

        return output_audios

        # audio_embeds = self.audio_model.encoder(input_audios).last_hidden_state

        # # audio_embeds = self.audio_mlp(audio_embeds)
        # audio_embeds = self.mlp2(audio_embeds)

        # output_audios = []
        # for i in range(input_audios.shape[0]):
        #     audio_span = input_audio_lengths[i][1]
        #     audio = audio_embeds[i][:audio_span]
        #     # if bos is not None:
        #     #     audio = torch.concat([bos, audio, eos])
        #     assert len(audio) == audio_span.item()
        #     output_audios.append(audio)

        # return output_audios

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep)[0].strip() for response in responses]
        return responses

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=False)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, ms_feat = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def chat_audio(self, tokenizer, pixel_values, audio_inputs, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            AUD_START_TOKEN='<audio>', AUD_END_TOKEN='</audio>', AUD_CONTEXT_TOKEN='<AUDIO_CONTEXT>',
            verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        aud_context_token_id = tokenizer.convert_tokens_to_ids(AUD_CONTEXT_TOKEN)
        self.aud_context_token_id = aud_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        if audio_inputs is not None:
            if self.config.qwa_version == "v1":
                audio_inputs_info = process_audio(audio_inputs)
            else:
                audio_inputs_info = process_audiov2(audio_inputs)
            num_audio_tokens = audio_inputs_info['audio_span_tokens'] - 2
            audio_tokens = AUD_START_TOKEN + AUD_CONTEXT_TOKEN * num_audio_tokens + AUD_END_TOKEN
            query = query.replace('<audio>', audio_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate_audio(
            pixel_values=pixel_values,
            input_audios=audio_inputs_info['input_audios'],
            input_audio_lengths=audio_inputs_info['input_audio_lengths'],
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate_audio(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_audios=None,
            input_audio_lengths=None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, ms_feat = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
            input_ids = input_ids.reshape(B, N)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        if input_audios is not None:
            B, L, C = input_embeds.shape

            if self.config.qwa_version == "v1":
                extract_audio_features = self.extract_audio_features_v1
            elif self.config.qwa_version == "v2":
                extract_audio_features = self.extract_audio_features_v2
            else:
                extract_audio_features = self.extract_audio_features_v2
            
            input_audios = torch.tensor(input_audios).cuda().unsqueeze(0).bfloat16()

            audio_embeds = extract_audio_features(input_audios, input_audio_lengths.unsqueeze(0))
            for i in range(B):
                audio_embed = audio_embeds[i]
                selected = (input_ids[i] == self.aud_context_token_id)
                assert sum(selected) == audio_embed.shape[0]
                input_embeds[i][selected] = input_embeds[i][selected] * 0.0 + audio_embed

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    
    def chat_mask(self, tokenizer, pixel_values, audio_inputs, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            AUD_START_TOKEN='<audio>', AUD_END_TOKEN='</audio>', AUD_CONTEXT_TOKEN='<AUDIO_CONTEXT>',
            verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        aud_context_token_id = tokenizer.convert_tokens_to_ids(AUD_CONTEXT_TOKEN)
        self.aud_context_token_id = aud_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        if audio_inputs is not None:
            if self.config.qwa_version == "v1":
                audio_inputs_info = process_audio(audio_inputs)
            else:
                audio_inputs_info = process_audiov2(audio_inputs)
            num_audio_tokens = audio_inputs_info['audio_span_tokens'] - 2
            audio_tokens = AUD_START_TOKEN + AUD_CONTEXT_TOKEN * num_audio_tokens + AUD_END_TOKEN
            query = query.replace('<audio>', audio_tokens, 1)
        else:
            audio_inputs_info = dict(input_audios=None, input_audio_lengths=None)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output, mask = self.generate_mask(
            pixel_values=pixel_values,
            input_audios=audio_inputs_info['input_audios'],
            input_audio_lengths=audio_inputs_info['input_audio_lengths'],
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history, mask
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response, mask

    @torch.no_grad()
    def generate_mask(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_audios=None,
            input_audio_lengths=None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, ms_feat = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
            input_ids = input_ids.reshape(B, N)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        if input_audios is not None:
            B, L, C = input_embeds.shape

            if self.config.qwa_version == "v1":
                extract_audio_features = self.extract_audio_features_v1
            elif self.config.qwa_version == "v2":
                extract_audio_features = self.extract_audio_features_v2
            else:
                extract_audio_features = self.extract_audio_features_v2
            
            input_audios = torch.tensor(input_audios).cuda().unsqueeze(0).bfloat16()

            audio_embeds = extract_audio_features(input_audios, input_audio_lengths.unsqueeze(0))
            for i in range(B):
                audio_embed = audio_embeds[i]
                selected = (input_ids[i] == self.aud_context_token_id)
                assert sum(selected) == audio_embed.shape[0]
                input_embeds[i][selected] = input_embeds[i][selected] * 0.0 + audio_embed

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=True,
            return_dict_in_generate=True,
            return_dict=return_dict,
            use_cache=True,
            # input_ids=input_ids,  # 3.5b模型加的
            **generate_kwargs,
        )
        
        masks = self.inference_mask(outputs, ms_feat)

        return outputs.sequences, masks
    
    @torch.no_grad()
    def inference_mask(self, outputs, ms_feat):
        # Get hidden states from the last layer
        # hidden_states is a tuple of length N (num_tokens), each containing all layer states
        # We need the last layer for each token
        hidden_states = torch.stack([token_states[-1] for token_states in outputs.hidden_states[1:]])
        token_id = outputs.sequences[:, 1:]
        
        # Find seg tokens in the sequence
        seg_token_mask = token_id == self.seg_token_id
        
        if not seg_token_mask.any():
            # No seg tokens found, return None
            return None
            
        # Get hidden states for seg tokens from the last layer
        seg_hidden_states = hidden_states[seg_token_mask[0]][0, 0]  # (N_seg, C)
        
        # Process ms_feat similar to mask_loss function
        # ms_feat shape is [T,C,H,W] where T is number of frames
        padded_ms_feats_list = [feat.unsqueeze(0) for feat in ms_feat]  # Add batch dim
        input_feats = {k: v for k, v in zip(["res2", "res3", "res4", "res5"], padded_ms_feats_list)}
        
        # Create attention mask for mask decoder - match number of frames
        T = ms_feat[0].shape[0]  # Get number of frames
        padded_mask = torch.ones((1, T), dtype=torch.bool, device=self.device)
        
        # Run mask decoder
        out = self.mask_decoder(input_feats, seg_hidden_states, padded_mask)
        
        # Get predicted masks
        pred_masks = out['pred_masks']  # Shape will be (N_seg, 1, H, W)
        # import torchshow as ts
        # ts.save(pred_masks.cpu().float() > 0.)
        
        return pred_masks
    
    def chat_mask_long_video(self, tokenizer, pixel_values, audio_inputs, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            AUD_START_TOKEN='<audio>', AUD_END_TOKEN='</audio>', AUD_CONTEXT_TOKEN='<AUDIO_CONTEXT>',
            verbose=False, frame_flag_list=None, num_image_token=None):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        # assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        aud_context_token_id = tokenizer.convert_tokens_to_ids(AUD_CONTEXT_TOKEN)
        self.aud_context_token_id = aud_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for i, num_patches in enumerate(num_patches_list):
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token[i] * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        if audio_inputs is not None:
            if self.config.qwa_version == "v1":
                audio_inputs_info = process_audio(audio_inputs)
            else:
                audio_inputs_info = process_audiov2(audio_inputs)
            num_audio_tokens = audio_inputs_info['audio_span_tokens'] - 2
            audio_tokens = AUD_START_TOKEN + AUD_CONTEXT_TOKEN * num_audio_tokens + AUD_END_TOKEN
            query = query.replace('<audio>', audio_tokens, 1)
        else:
            audio_inputs_info = dict(input_audios=None, input_audio_lengths=None)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output, mask = self.generate_mask_long_video(
            pixel_values=pixel_values,
            input_audios=audio_inputs_info['input_audios'],
            input_audio_lengths=audio_inputs_info['input_audio_lengths'],
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_flag_list=frame_flag_list,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history, mask
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response, mask

    @torch.no_grad()
    def generate_mask_long_video(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_audios=None,
            input_audio_lengths=None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            frame_flag_list=None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                # Process video frames in clips to save memory
                clip_size = 10  # Process 32 frames at a time
                vit_embeds_list = []
                ms_feat_list = []
                
                for i in range(0, pixel_values.shape[0], clip_size):
                    clip = pixel_values[i:i+clip_size]
                    clip_embeds, clip_ms_feat = self.extract_feature(clip)
                    vit_embeds_list.append(clip_embeds)
                    ms_feat_list.append(clip_ms_feat)
                    
                # Concatenate results from all clips
                vit_embeds = torch.cat(vit_embeds_list, dim=0)
                ms_feat = []
                for i in range(len(ms_feat_list[0])):
                    ms_feat.append(torch.cat([clip_ms_feat[i] for clip_ms_feat in ms_feat_list], dim=0))
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            # if frame_flag_list is not None:
            #     frame_flag_list = torch.tensor(frame_flag_list).long().to(self.device)
            
            T, num_tokens, C = vit_embeds.shape
            pooled_embeds = []
            for i in range(T):
                if frame_flag_list[i] == 1:
                    # Average pool the 256 tokens into 1 token
                    pooled_token = vit_embeds[i].mean(dim=0, keepdim=True) # [1, C]
                    pooled_embeds.append(pooled_token)
                elif frame_flag_list[i] == 2:
                    # Keep all 256 tokens
                    pooled_embeds.append(vit_embeds[i])
                else:
                    continue
            resample_vit_embeds = torch.cat(pooled_embeds, dim=0)
            
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = resample_vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
            input_ids = input_ids.reshape(B, N)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        if input_audios is not None:
            B, L, C = input_embeds.shape

            if self.config.qwa_version == "v1":
                extract_audio_features = self.extract_audio_features_v1
            elif self.config.qwa_version == "v2":
                extract_audio_features = self.extract_audio_features_v2
            else:
                extract_audio_features = self.extract_audio_features_v2
            
            input_audios = torch.tensor(input_audios).cuda().unsqueeze(0).bfloat16()

            audio_embeds = extract_audio_features(input_audios, input_audio_lengths.unsqueeze(0))
            for i in range(B):
                audio_embed = audio_embeds[i]
                selected = (input_ids[i] == self.aud_context_token_id)
                assert sum(selected) == audio_embed.shape[0]
                input_embeds[i][selected] = input_embeds[i][selected] * 0.0 + audio_embed

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=True,
            return_dict_in_generate=True,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )
        
        masks = self.inference_mask(outputs, ms_feat)

        return outputs.sequences, masks

    
    def chat_omniavs(self, tokenizer, pixel_values, audio_inputs, question, generation_config, history=None, return_history=False,
            num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
            AUD_START_TOKEN='<audio>', AUD_END_TOKEN='</audio>', AUD_CONTEXT_TOKEN='<AUDIO_CONTEXT>',
            verbose=False, frame_flag_list=None, num_image_token=None):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        # assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        aud_context_token_id = tokenizer.convert_tokens_to_ids(AUD_CONTEXT_TOKEN)
        self.aud_context_token_id = aud_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for i, num_patches in enumerate(num_patches_list):
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token[i] * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        if audio_inputs is not None:
            audio_input_list = []
            input_audio_lengths = []
            for audio_path in audio_inputs:
                audio_info = process_audiov2(audio_path)
                audio_input_list.append(torch.tensor(audio_info['input_audios']))
                input_audio_lengths.append(audio_info['input_audio_lengths'])
         
            for i in range(len(input_audio_lengths)):
                audio_tokens = AUD_START_TOKEN + AUD_CONTEXT_TOKEN * input_audio_lengths[i][-1] + AUD_END_TOKEN
                query = query.replace('<audio>', audio_tokens, 1)
                
            audio_inputs_info = dict(input_audios=torch.stack(audio_input_list, dim=0), input_audio_lengths=torch.stack(input_audio_lengths, dim=0))
        else:
            audio_inputs_info = dict(input_audios=None, input_audio_lengths=None)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output, mask = self.generate_omniavs(
            pixel_values=pixel_values,
            input_audios=audio_inputs_info['input_audios'],
            input_audio_lengths=audio_inputs_info['input_audio_lengths'],
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_flag_list=frame_flag_list,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history, mask
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response, mask

    @torch.no_grad()
    def generate_omniavs(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_audios=None,
            input_audio_lengths=None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            frame_flag_list=None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                # Process video frames in clips to save memory
                clip_size = 10  # Process 32 frames at a time
                vit_embeds_list = []
                ms_feat_list = []
                
                for i in range(0, pixel_values.shape[0], clip_size):
                    clip = pixel_values[i:i+clip_size]
                    clip_embeds, clip_ms_feat = self.extract_feature(clip)
                    vit_embeds_list.append(clip_embeds)
                    ms_feat_list.append(clip_ms_feat)
                    
                # Concatenate results from all clips
                vit_embeds = torch.cat(vit_embeds_list, dim=0)
                ms_feat = []
                for i in range(len(ms_feat_list[0])):
                    ms_feat.append(torch.cat([clip_ms_feat[i] for clip_ms_feat in ms_feat_list], dim=0))
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            # if frame_flag_list is not None:
            #     frame_flag_list = torch.tensor(frame_flag_list).long().to(self.device)
            
            T, num_tokens, C = vit_embeds.shape
            pooled_embeds = []
            for i in range(T):
                if frame_flag_list[i] == 1:
                    # Average pool the 256 tokens into 1 token
                    pooled_token = vit_embeds[i].mean(dim=0, keepdim=True) # [1, C]
                    pooled_embeds.append(pooled_token)
                elif frame_flag_list[i] == 2:
                    # Keep all 256 tokens
                    pooled_embeds.append(vit_embeds[i])
                else:
                    continue
            resample_vit_embeds = torch.cat(pooled_embeds, dim=0)
            
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = resample_vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
            input_ids = input_ids.reshape(B, N)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        if input_audios is not None:
            B, L, C = input_embeds.shape

            if self.config.qwa_version == "v1":
                extract_audio_features = self.extract_audio_features_v1
            elif self.config.qwa_version == "v2":
                extract_audio_features = self.extract_audio_features_v2
            else:
                extract_audio_features = self.extract_audio_features_v2
            
            input_audios = torch.tensor(input_audios).cuda().bfloat16()

            audio_embeds = extract_audio_features(input_audios, input_audio_lengths)
            
            audio_embeds = torch.cat(audio_embeds, dim=0)
            selected = (input_ids[0] == self.aud_context_token_id)
            assert sum(selected) == audio_embeds.shape[0]
            input_embeds[0][selected] = input_embeds[0][selected] * 0.0 + audio_embeds
            # for i in range(B):
            #     audio_embed = audio_embeds[i]
            #     selected = (input_ids[i] == self.aud_context_token_id)
            #     assert sum(selected) == audio_embed.shape[0]
            #     input_embeds[i][selected] = input_embeds[i][selected] * 0.0 + audio_embed

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=True,
            return_dict_in_generate=True,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )
        
        masks = self.inference_mask(outputs, ms_feat)

        return outputs.sequences, masks
    
    
    def chat_omniavs_new(self, tokenizer, pixel_values, audio_inputs, question, generation_config, history=None, return_history=False,
        num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        AUD_START_TOKEN='<aud>', AUD_END_TOKEN='</aud>', AUD_CONTEXT_TOKEN='<AUDIO_CONTEXT>',
        SUB_AUDIO_START_TOKEN='<sub_aud>', SUB_AUDIO_END_TOKEN='</sub_aud>', SUB_AUDIO_CONTEXT_TOKEN='<SUB_AUDIO_CONTEXT>',
        verbose=False, frame_flag_list=None, num_image_token=None):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        # assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        aud_context_token_id = tokenizer.convert_tokens_to_ids(AUD_CONTEXT_TOKEN)
        self.aud_context_token_id = aud_context_token_id

        sub_audio_context_token_id = tokenizer.convert_tokens_to_ids(SUB_AUDIO_CONTEXT_TOKEN)
        self.sub_audio_context_token_id = sub_audio_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep)

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for i, num_patches in enumerate(num_patches_list):
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token[i] * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
        
        if audio_inputs is not None:
            audio_input_list = []
            input_audio_lengths = []
            for audio_path in audio_inputs:
                audio_info = process_audiov2(audio_path)
                audio_input_list.append(torch.tensor(audio_info['input_audios']))
                input_audio_lengths.append(audio_info['input_audio_lengths'])

            if "sub_audio" in query:
                # Only use first audio input for sub_audio tokens
                
                total_length = input_audio_lengths[0][-1]
                num_segments = query.count('<sub_audio>')
                base_length = total_length // num_segments
                remainder = total_length % num_segments
                
                for i in range(num_segments):
                    # Add one extra token for the first 'remainder' segments
                    current_length = base_length + (1 if i < remainder else 0)
                    audio_tokens = SUB_AUDIO_START_TOKEN + SUB_AUDIO_CONTEXT_TOKEN * current_length + SUB_AUDIO_END_TOKEN
                    query = query.replace('<sub_audio>', audio_tokens, 1)
                    
            for i in range(len(input_audio_lengths)):
                audio_tokens = AUD_START_TOKEN + AUD_CONTEXT_TOKEN * input_audio_lengths[i][-1] + AUD_END_TOKEN
                query = query.replace('<audio>', audio_tokens, 1)
                
            audio_inputs_info = dict(input_audios=torch.stack(audio_input_list, dim=0), input_audio_lengths=torch.stack(input_audio_lengths, dim=0))
        else:
            audio_inputs_info = dict(input_audios=None, input_audio_lengths=None)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].cuda()
        attention_mask = model_inputs['attention_mask'].cuda()
        generation_config['eos_token_id'] = eos_token_id
        generation_output, mask = self.generate_omniavs_new(
            pixel_values=pixel_values,
            input_audios=audio_inputs_info['input_audios'],
            input_audio_lengths=audio_inputs_info['input_audio_lengths'],
            input_ids=input_ids,
            attention_mask=attention_mask,
            frame_flag_list=frame_flag_list,
            **generation_config
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep)[0].strip()
        history.append((question, response))
        if return_history:
            return response, history, mask
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response, mask

    @torch.no_grad()
    def generate_omniavs_new(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_audios=None,
            input_audio_lengths=None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            frame_flag_list=None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                # Process video frames in clips to save memory
                clip_size = 10  # Process 32 frames at a time
                vit_embeds_list = []
                ms_feat_list = []
                
                for i in range(0, pixel_values.shape[0], clip_size):
                    clip = pixel_values[i:i+clip_size]
                    clip_embeds, clip_ms_feat = self.extract_feature(clip)
                    vit_embeds_list.append(clip_embeds)
                    ms_feat_list.append(clip_ms_feat)
                    
                # Concatenate results from all clips
                vit_embeds = torch.cat(vit_embeds_list, dim=0)
                ms_feat = []
                for i in range(len(ms_feat_list[0])):
                    ms_feat.append(torch.cat([clip_ms_feat[i] for clip_ms_feat in ms_feat_list], dim=0))
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            # if frame_flag_list is not None:
            #     frame_flag_list = torch.tensor(frame_flag_list).long().to(self.device)
            
            T, num_tokens, C = vit_embeds.shape
            pooled_embeds = []
            for i in range(T):
                if frame_flag_list[i] == 1:
                    # Average pool the 256 tokens into 1 token
                    pooled_token = vit_embeds[i].mean(dim=0, keepdim=True) # [1, C]
                    pooled_embeds.append(pooled_token)
                elif frame_flag_list[i] == 2:
                    # Keep all 256 tokens
                    pooled_embeds.append(vit_embeds[i])
                else:
                    continue
            resample_vit_embeds = torch.cat(pooled_embeds, dim=0)
            
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = resample_vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
            input_ids = input_ids.reshape(B, N)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        if input_audios is not None:
            B, L, C = input_embeds.shape

            if self.config.qwa_version == "v1":
                extract_audio_features = self.extract_audio_features_v1
            elif self.config.qwa_version == "v2":
                extract_audio_features = self.extract_audio_features_v2
            else:
                extract_audio_features = self.extract_audio_features_v2
            
            input_audios = torch.tensor(input_audios).cuda().bfloat16()

            audio_embeds_list = extract_audio_features(input_audios, input_audio_lengths)
            
            audio_embeds = torch.cat(audio_embeds_list, dim=0)
            selected = (input_ids[0] == self.aud_context_token_id)
            assert sum(selected) == audio_embeds.shape[0]
            input_embeds[0][selected] = input_embeds[0][selected] * 0.0 + audio_embeds
            
            filtered_sub_audio_embeds = [audio_embeds_list[0]]
            if len(filtered_sub_audio_embeds) > 0:
                sub_audio_embeds = torch.cat(filtered_sub_audio_embeds, dim=0)
                sub_audio_selected = (input_ids == self.sub_aud_context_token_id)
                input_embeds[sub_audio_selected] = input_embeds[sub_audio_selected] * 0.0 + sub_audio_embeds
            # for i in range(B):
            #     audio_embed = audio_embeds[i]
            #     selected = (input_ids[i] == self.aud_context_token_id)
            #     assert sum(selected) == audio_embed.shape[0]
            #     input_embeds[i][selected] = input_embeds[i][selected] * 0.0 + audio_embed

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=True,
            return_dict_in_generate=True,
            return_dict=return_dict,
            use_cache=True,
            **generate_kwargs,
        )
        
        masks = self.inference_mask(outputs, ms_feat)

        return outputs.sequences, masks