# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY, build_sem_seg_head, ShapeSpec
from detectron2.config import get_cfg
from detectron2.projects.point_rend.point_features import (
    # get_uncertain_point_coords_with_randomness,
    point_sample,
)

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import build_transformer_decoder
from mask2former.modeling.pixel_decoder.fpn import build_pixel_decoder
from mask2former.config import add_maskformer2_config
from mask2former.modeling.criterion import dice_loss_jit, sigmoid_ce_loss_jit


def setup():
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_maskformer2_config(cfg)
    cfg.merge_from_file("mask2former/configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml")
    cfg.freeze()
    return cfg


@SEM_SEG_HEADS_REGISTRY.register()
class LLMMaskDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.debug(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
        transformer_in_feature: str,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor
        self.transformer_in_feature = transformer_in_feature

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        if cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "transformer_encoder":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "pixel_embedding":
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        elif cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE == "multi_scale_pixel_decoder":  # for maskformer2
            transformer_predictor_in_channels = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        else:
            transformer_predictor_in_channels = input_shape[cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE].channels

        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_in_feature": cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE,
            "transformer_predictor": build_transformer_decoder(
                cfg,
                transformer_predictor_in_channels,
                mask_classification=True,
            ),
        }

    def forward(self, features, seg_hidden_states, padded_mask, mask=None):
        return self.layers(features, seg_hidden_states, padded_mask, mask)

    def layers(self, features, seg_hidden_states, padded_mask, mask=None):
        bs, T = padded_mask.shape

        ms_features = {k: v.flatten(0, 1) for k, v in features.items()}

        # pixel_deocder
        if self.training:
            mask_features, transformer_encoder_features, multi_scale_features = self.pixel_decoder.forward_features(ms_features)
        else:
            # Process features in clips to save memory
            clip_size = 10  # Process 10 frames at a time
            mask_features_list = []
            transformer_encoder_features_list = []
            multi_scale_features_list = []
            
            for i in range(0, len(next(iter(ms_features.values()))), clip_size):
                # Get clip of features
                clip_features = {k: v[i:i+clip_size] for k, v in ms_features.items()}
                
                # Process clip
                clip_mask_features, clip_transformer_features, clip_multi_scale = self.pixel_decoder.forward_features(clip_features)
                
                # Store results
                mask_features_list.append(clip_mask_features)
                transformer_encoder_features_list.append(clip_transformer_features)
                
                # Handle multi-scale features for first clip
                if i == 0:
                    multi_scale_features_list = [[] for _ in clip_multi_scale]
                
                # Store multi-scale features
                for j, feat in enumerate(clip_multi_scale):
                    multi_scale_features_list[j].append(feat)
                    
            # Concatenate results from all clips
            mask_features = torch.cat(mask_features_list, dim=0)
            transformer_encoder_features = torch.cat(transformer_encoder_features_list, dim=0)
            multi_scale_features = [torch.cat(feat_list, dim=0) for feat_list in multi_scale_features_list]

        # mask_features = mask_features.view(bs, T, *mask_features.shape[-3:])
        # multi_scale_features = [feat.view(bs, T, *feat.shape[-3:]) for feat in multi_scale_features]

        # mask_decoder
        predictions = self.predictor(multi_scale_features, mask_features, seg_hidden_states, padded_mask, mask)
        return predictions
    

from mask2former.modeling.transformer_decoder.maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY
from mask2former.modeling.transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder, SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP
from mask2former.modeling.transformer_decoder.position_encoding import PositionEmbeddingSine

@TRANSFORMER_DECODER_REGISTRY.register()
class LLMMultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        seg_token_dim: int,
        use_ps: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            # self.transformer_self_attention_layers.append(
            #     SelfAttentionLayer(
            #         d_model=hidden_dim,
            #         nhead=nheads,
            #         dropout=0.0,
            #         normalize_before=pre_norm,
            #     )
            # )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # self.seg_token_proj = MLP(4096, 512, 512, 2)
        self.seg_token_proj = MLP(seg_token_dim, 512, 512, 2)
        # self.seg_token_proj = MLP(3072, 512, 512, 2)
        
        self.hidden_dim = hidden_dim
        self.use_ps = use_ps
        if use_ps:
            self.mask_feature_ps_mlp = MLP(hidden_dim // 4, hidden_dim, hidden_dim, 2)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        
        ret["seg_token_dim"] = cfg.MODEL.SEM_SEG_HEAD.SEG_TOKEN_DIM
        ret["use_ps"] = cfg.MODEL.SEM_SEG_HEAD.USE_PS

        return ret

    def forward(self, x, mask_features, seg_hidden_states, padded_mask, mask = None):
        # x is a list of multi-scale feature
        # bs, T = mask_features.shape[:3]
        mask_features = mask_features.to(seg_hidden_states.dtype)
        x = [_.to(seg_hidden_states.dtype) for _ in x]
        # x
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2).to(seg_hidden_states.dtype))
            src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # # flatten NxCxHxW to HWxNxC
            # pos[-1] = pos[-1].permute(2, 0, 1)
            # src[-1] = src[-1].permute(2, 0, 1)

        bs, T = padded_mask.shape

        # online的形式
        # _, bs, _ = src[0].shape

        # # query_embed and query_feat
        # seg_token_id = 1000
        # seg_token_mask = input_ids == seg_token_id
        # seg_hidden_states = hidden_states[seg_token_mask]   # N_t, 786

        seg_token_feat = self.seg_token_proj(seg_hidden_states)
        query_embed = seg_token_feat[:, :self.hidden_dim].unsqueeze(0)  # L x B x 256 | (1, 3, 256)
        output = seg_token_feat[:, self.hidden_dim:].unsqueeze(0)  # # L x B x 256 | (1, 3, 256)

        # 重新整理mask features，pos和src
        mask_features = mask_features.view(bs, T, *mask_features.shape[-3:])
        
        if self.use_ps:
            # Pixel unshuffle to increase spatial dimensions by 2x
            B, T, C, H, W = mask_features.shape
            mask_features = mask_features.view(B*T, C, H, W)
            mask_features = F.pixel_unshuffle(mask_features, 2)  # Increases spatial dims by 2x
            mask_features = mask_features.view(B, T, -1, H*2, W*2)
            # mask_features shape: [2, 10, 64, 224, 224]
            B, T, C, H, W = mask_features.shape
            mask_features = mask_features.view(B*T, C,H, W)
            mask_features = mask_features.permute(0, 2, 3, 1)
            mask_features = self.mask_feature_ps_mlp(mask_features)
            mask_features = mask_features.permute(0, 3, 1, 2)
            mask_features = mask_features.view(B, T, -1, H, W)

        src_list = []
        pos_list = []
        for t in range(T):
            src_list.append(
                [src_.reshape(bs, T, *src_.shape[-2:])[:, t].permute(2, 0, 1).contiguous() for src_ in src]
            )
            pos_list.append(
                [pos_.reshape(bs, T, *pos_.shape[-2:])[:, t].permute(2, 0, 1).contiguous() for pos_ in pos]
            )
        
        predictions_class_list = []
        predictions_mask_list = []

        for t in range(T):
            # online的形式
            _predictions_class = []
            _predictions_mask = []
            _mask_features = mask_features[:, t]
            _src = src_list[t]
            _pos = pos_list[t]
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, _mask_features, attn_mask_target_size=size_list[0])
            _predictions_class.append(outputs_class)  # todo: 有mask为1，无mask为0
            _predictions_mask.append(outputs_mask)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, _src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=_pos[level_index], query_pos=query_embed
                )

                output = self.transformer_ffn_layers[i](
                    output
                )

                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, _mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
                _predictions_class.append(outputs_class)
                _predictions_mask.append(outputs_mask)
            
            assert len(_predictions_class) == self.num_layers + 1

            predictions_class_list.append(_predictions_class[-1])
            predictions_mask_list.append(_predictions_mask[-1])
        
        pred_masks = torch.cat(predictions_mask_list, dim=1)
        pred_logits = torch.cat(predictions_class_list, dim=1)

        out = {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks

        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]


def loss_labels(outputs, targets):
    empty_weight = torch.ones(2, device=targets.device)
    empty_weight[-1] = 0.1
    empty_weight = empty_weight.to(outputs.dtype)
    loss_ce = F.cross_entropy(outputs, targets, empty_weight)
    return loss_ce

from detectron2.layers import cat, shapes_to_tensor

def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device).to(coarse_logits.dtype)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))

def loss_masks(src_masks, target_masks, num_masks):
    """Compute the losses related to the masks: the focal loss and the dice loss.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """
    # assert "pred_masks" in outputs

    # src_idx = self._get_src_permutation_idx(indices)
    # tgt_idx = self._get_tgt_permutation_idx(indices)
    # src_masks = outputs["pred_masks"]
    # src_masks = src_masks[src_idx]
    # masks = [t["masks"] for t in targets]
    # # TODO use valid to mask invalid areas due to padding in loss
    # target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
    # target_masks = target_masks.to(src_masks)
    # target_masks = target_masks[tgt_idx]

    # No need to upsample predictions as we are using normalized coordinates :)
    # N x 1 x H x W
    src_masks = src_masks[:, None]
    target_masks = target_masks[:, None]

    with torch.no_grad():
        # sample point_coords
        point_coords = get_uncertain_point_coords_with_randomness(
            src_masks,
            lambda logits: calculate_uncertainty(logits),
            12544,
            3.0,
            0.75,
        ).to(src_masks.dtype)
        # get gt labels
        point_labels = point_sample(
            target_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1).to(src_masks.dtype)

    point_logits = point_sample(
        src_masks,
        point_coords,
        align_corners=False,
    ).squeeze(1)

    losses = {
        "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
        "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
    }

    del src_masks
    del target_masks
    return losses["loss_mask"], losses['loss_dice']


def mask_loss(ms_feat, input_ids, hidden_states, image_num, masks):
    ms_feat = list(ms_feat.values())
    assert ms_feat[0].shape[0] == torch.sum(image_num), "Wrong: Number of image!!!"
    seg_token_id = 1000
    seg_token_mask = input_ids == seg_token_id
    
    # 计算每个样本中 seg_token 的数量
    seg_token_counts = seg_token_mask.sum(dim=1)

    # 计算累计图片数量，用于索引 ms_feat
    cumulative_image_nums = torch.cat([torch.tensor([0]).to(seg_token_counts.device), image_num.cumsum(dim=0)]).to(seg_token_counts.device)

    # 获取所有 seg_token 的数量
    total_seg_tokens = seg_token_counts.sum().item()

    # 初始化一个用于存储最终特征的 tensor
    max_num_images = image_num.max().item()
    # 生成一个索引列表，表示每个 seg_token 对应的 ms_feat 起始索引
    token_indices = torch.repeat_interleave(cumulative_image_nums[:-1], seg_token_counts)
    
    # 获取每个 seg_token 对应的填充长度
    padding_lengths = image_num.repeat_interleave(seg_token_counts)
    padding_sizes = max_num_images - padding_lengths
    padded_ms_feats_list = list()
    for ms_feat_ in ms_feat:
        padded_ms_feats = torch.zeros((total_seg_tokens, max_num_images, *ms_feat_.shape[-3:])).to(seg_token_counts.device)
        # padded_ms_feats_list.append(padded_ms_feats)

        # 生成每个 seg_token 对应的 ms_feat 特征图
        ms_feat_for_tokens = torch.cat([ms_feat_[token_indices[i]:token_indices[i] + image_num[i]]
                                        for i in range(len(image_num))
                                        if seg_token_counts[i] > 0])
        
        # 对 ms_feat 进行填充
        for i in range(total_seg_tokens):
            padded_ms_feats[i, :padding_lengths[i]] = ms_feat_for_tokens[i]
        padded_ms_feats_list.append(padded_ms_feats)

    # 生成 padded_mask
    padded_mask = torch.zeros((total_seg_tokens, max_num_images), dtype=torch.bool)

    for i in range(total_seg_tokens):
        padded_mask[i, :padding_lengths[i]] = 1
    
    input_feats = {
        k: v for k, v in zip(["res2", "res3", "res4", "res5"], padded_ms_feats_list)
    }

    out = model(input_feats, input_ids, hidden_states, padded_mask)

    return out    


if __name__ == "__main__":
    pass
    cfg = setup()
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
    model = build_sem_seg_head(cfg, shape_list).eval().cuda()
    print("here")
    input = {
        "res2": torch.randn((10, 1024, 112, 112), device="cuda"),
        "res3": torch.randn((10, 1024, 56, 56), device="cuda"),
        "res4": torch.randn((10, 1024, 28, 28), device="cuda"),
        "res5": torch.randn((10, 1024, 14, 14), device="cuda"),
    }

    hidden_states = torch.randn((4, 4096, 786), device="cuda")
    input_ids = torch.zeros((4, 4096), device="cuda")
    input_ids[1, 3] = 1000
    input_ids[1, 10] = 1000
    input_ids[2, 6] = 1000
    image_num = torch.tensor([5, 4, 1, 0], device='cuda')
    masks = torch.tensor([4, 448, 448])
    
    out = mask_loss(input, input_ids, hidden_states, image_num, masks)
