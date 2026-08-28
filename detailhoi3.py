"""
CJ
ho_q:                            
q_pos: skeleton (23) / object - n1:n2 - n1+n2=512
"""

from __future__ import annotations
import os
import torch
import torch.nn.functional as F
import torch.distributed as dist

from torch import nn, Tensor
from collections import OrderedDict
from typing import Optional, Tuple, List
from torchvision.ops import FeaturePyramidNetwork

from transformers import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerDecoderLayer,
    SwinTransformer,
)

from ops import (
    binary_focal_loss_with_logits,
    compute_spatial_encodings,
    prepare_region_proposals,
    associate_with_ground_truth,
    compute_prior_scores,
    compute_sinusoidal_pe
)

from detr.models import build_model as build_base_detr
from detr.models.position_encoding import PositionEmbeddingSine
from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list

from build_model_pose_17 import BodyPoseEstimator, KeypointEstimator

# ── 修改这两个值来控制 centre_feat 的两段维度，总和必须等于 kv_dim*2=512 ──
HUMAN_CENTRE_DIM = 256
OBJECT_CENTRE_DIM = 256

# 关键点总数（与 kpts_pe_proj 输入维度一致）
TOTAL_KEYPOINTS = 23


class MultiModalFusion(nn.Module):
    def __init__(self, fst_mod_size, scd_mod_size, repr_size):
        super().__init__()
        self.fc1 = nn.Linear(fst_mod_size, repr_size)
        self.fc2 = nn.Linear(scd_mod_size, repr_size)
        self.ln1 = nn.LayerNorm(repr_size)
        self.ln2 = nn.LayerNorm(repr_size)

        mlp = []
        repr_size_list = [2 * repr_size, int(repr_size * 1.5), repr_size]
        for d_in, d_out in zip(repr_size_list[:-1], repr_size_list[1:]):
            mlp.append(nn.Linear(d_in, d_out))
            mlp.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        x = self.ln1(self.fc1(x))
        y = self.ln2(self.fc2(y))
        z = F.relu(torch.cat([x, y], dim=-1))
        z = self.mlp(z)
        return z


# ============================================================================
#                              HumanObjectMatcher
# ============================================================================
class HumanObjectMatcher(nn.Module):
    def __init__(self, repr_size, num_verbs, human_dim=HUMAN_CENTRE_DIM,
                 obj_dim=OBJECT_CENTRE_DIM, obj_to_verb=None, dropout=.1, human_idx=0):
        super().__init__()
        self.repr_size = repr_size
        self.num_verbs = num_verbs
        self.human_idx = human_idx
        self.obj_to_verb = obj_to_verb
        self.human_centre_dim = human_dim    # human skeleton PE 输出维度
        self.object_centre_dim = obj_dim     # object centre PE 输出维度

        # 用于对 bbox 的中心 / 尺度做条件调节
        self.ref_anchor_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.ref_keypoint_head = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )
        # 空间编码 → repr_size
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),  nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, repr_size), nn.ReLU(),
        )

        # sinusoidal PE 投影:
        # human: TOTAL_KEYPOINTS(17) × 256 → human_centre_dim
        self.kpts_pe_proj = nn.Linear(TOTAL_KEYPOINTS * 256, self.human_centre_dim)
        # object: 256 → object_centre_dim
        self.obj_c_pe_proj = nn.Linear(256, self.object_centre_dim)  # Bug1修复：补充缺失定义

        # 对 region-level embedding 做一个浅层 encoder
        self.encoder = TransformerEncoder(num_layers=2, dropout=dropout)

        # multimodal fusion: [human,obj] visual (512) + spatial (repr_size)
        self.mmf = MultiModalFusion(512, repr_size, repr_size)

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        nh = torch.sum(is_human)
        if not torch.all(labels[:nh] == self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return nh

    def compute_box_pe(self, boxes, embeds, body_kpts, image_size):
            """
            body_kpts: [N, 17, 3] (原始输入为17个点)
            returns:
                box_pe:      [N, 512]
                c_pe_out:    [N, object_centre_dim]
                kpts_pe_out: [N, human_centre_dim] (基于23个点的投影)
            """
            # --- 新增：关键点插值逻辑 (17 -> 23) ---
            # COCO 17 索引参考:
            # 5:左肩, 6:右肩, 7:左肘, 8:右肘, 11:左髋, 12:右髋, 13:左膝, 14:右膝
            connection_indices = [
                (5, 7),   # 左肩 ↔ 左肘
                (6, 8),   # 右肩 ↔ 右肘
                (11, 13), # 左髋 ↔ 左膝
                (12, 14), # 右髋 ↔ 右膝
                (5, 6),   # 左肩 ↔ 右肩
                (11, 12)  # 左髋 ↔ 右髋
            ]
            
            N = body_kpts.shape[0]
            device = body_kpts.device
            
            # 计算中点坐标和置信度（取两个端点置信度的最小值，确保插值点在两个端点都可靠时才有效）
            interp_kpts = []
            for idx1, idx2 in connection_indices:
                p1 = body_kpts[:, idx1, :2]
                p2 = body_kpts[:, idx2, :2]
                conf1 = body_kpts[:, idx1, 2:3]
                conf2 = body_kpts[:, idx2, 2:3]
                
                mid_p = (p1 + p2) / 2.0
                mid_conf = torch.min(conf1, conf2)
                interp_kpts.append(torch.cat([mid_p, mid_conf], dim=-1).unsqueeze(1))
                
            # 拼接原始 17 个点和 6 个插值点 -> [N, 23, 3]
            enhanced_kpts = torch.cat([body_kpts] + interp_kpts, dim=1)
            
            # --- 后续逻辑使用 enhanced_kpts (23个点) ---
            bx_norm = boxes / image_size[[1, 0, 1, 0]]
            bx_c = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
            b_wh = bx_norm[:, 2:] - bx_norm[:, :2]

            kpts_norm = enhanced_kpts.clone() # 使用 23 个点
            kpts_norm[..., 0] /= image_size[1]   # x / W
            kpts_norm[..., 1] /= image_size[0]   # y / H
            kpts_xy = kpts_norm[..., :2]          # [N, 23, 2]

            c_pe  = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)   # [N, 256]
            wh_pe = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)   # [N, 256]
            kpts_pe = compute_sinusoidal_pe(kpts_xy, 20)                   # [N, 23, 256]

            box_pe = torch.cat([c_pe, wh_pe], dim=-1)                      # [N, 512]
            ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()

            c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
            c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

            kpts_conf = enhanced_kpts[..., 2]          # [N, 23]
            valid_mask = kpts_conf > 0.3

            kpts_wh = torch.zeros(N, 2, device=device)
            for i in range(N):
                valid_kpts_i = kpts_xy[i][valid_mask[i]]
                if len(valid_kpts_i) > 0:
                    kpts_min = valid_kpts_i.min(0)[0]
                    kpts_max = valid_kpts_i.max(0)[0]
                    kpts_wh[i] = kpts_max - kpts_min
                else:
                    kpts_wh[i] = torch.ones(2, device=device)

            ref_kpts_wh = self.ref_keypoint_head(embeds).sigmoid()        # [N, 2]
            kpts_scale  = ref_kpts_wh / (kpts_wh + 1e-6)

            kpts_pe[..., :128] *= kpts_scale[:, 1].unsqueeze(-1).unsqueeze(-1)
            kpts_pe[..., 128:] *= kpts_scale[:, 0].unsqueeze(-1).unsqueeze(-1)

            # 投影到目标维度：此时 TOTAL_KEYPOINTS=23，kpts_pe.reshape(N, -1) 维度为 [N, 23*256]
            kpts_pe_out = self.kpts_pe_proj(kpts_pe.reshape(N, -1))   # [N, human_centre_dim]
            c_pe_out    = self.obj_c_pe_proj(c_pe)                    

            return box_pe, c_pe_out, kpts_pe_out

    # ------------------------------------------------------------------------
    # MAIN FORWARD
    # ------------------------------------------------------------------------
    def forward(self, region_props, image_sizes, device=None):
        """
        region_props : list[dict]  每张图的 proposal 字典
        image_sizes  : Tensor [B, 2]  (H, W)
        """
        if device is None:
            device = region_props[0]["hidden_states"].device

        ho_queries        = []
        paired_indices    = []
        prior_scores      = []
        object_types      = []
        positional_embeds = []

        for i, rp in enumerate(region_props):
            boxes     = rp["boxes"]
            scores    = rp["scores"]
            labels    = rp["labels"]
            embeds    = rp["hidden_states"]
            body_kpts = rp["keypoints"]    # (n, TOTAL_KEYPOINTS, 3)

            nh = self.check_human_instances(labels)
            n  = len(boxes)

            # ---------------- 枚举 HO pairs ----------------
            # Bug5修复：添加 indexing='ij' 避免 PyTorch 版本警告/错误
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device),
            )
            x_keep, y_keep = torch.nonzero(
                torch.logical_and(x != y, x < nh)
            ).unbind(1)

            if len(x_keep) == 0:
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                # Bug6修复：空 pair 时补充结构完整的 positional_embeds，避免 decoder 访问时 KeyError
                positional_embeds.append({
                    "centre": torch.zeros(0, 1, self.human_centre_dim + self.object_centre_dim, device=device),
                    "box":    torch.zeros(0, 1, 1024, device=device),  # box_pe dim = 512*2
                })
                continue

            # ---------------- 空间编码 ----------------
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x.flatten()],], [boxes[y.flatten()],], [image_sizes[i],]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            pairwise_spatial = pairwise_spatial.reshape(n, n, -1)

            # ---------------- positional encoding ----------------
            box_pe, c_pe, kpts_pe_out = self.compute_box_pe(
                boxes, embeds, body_kpts, image_sizes[i]
            )
            embeds, _ = self.encoder(embeds.unsqueeze(1), box_pe.unsqueeze(1))
            embeds = embeds.squeeze(1)  # (n, 256)

            # ---------------- HO query (内容特征) ----------------
            ho_q = self.mmf(
                torch.cat([embeds[x_keep], embeds[y_keep]], dim=1),  # (num_pairs, 512)
                pairwise_spatial[x_keep, y_keep]                     # (num_pairs, repr_size)
            )

            ho_queries.append(ho_q)
            paired_indices.append(torch.stack([x_keep, y_keep], dim=1))
            prior_scores.append(compute_prior_scores(
                x_keep, y_keep, scores, labels,
                self.num_verbs, self.training, self.obj_to_verb
            ))
            object_types.append(labels[y_keep])

            # Bug2修复：直接使用 compute_box_pe 已返回的 kpts_pe_out，不再重复投影
            # ---------------- positional embeds for decoder ----------------
            centre_feat = torch.cat([
                kpts_pe_out[x_keep],  # human keypoint PE (human_centre_dim=256)
                c_pe[y_keep],         # object center PE  (object_centre_dim=256)
            ], dim=-1)                # (num_pairs, 512)

            positional_embeds.append({
                "centre": centre_feat.unsqueeze(1),                                          # (num_pairs, 1, 512)
                "box":    torch.cat([box_pe[x_keep], box_pe[y_keep]], dim=-1).unsqueeze(1)   # (num_pairs, 1, 1024)
            })

        return ho_queries, paired_indices, prior_scores, object_types, positional_embeds


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(self.dims)


class FeatureHead(nn.Module):
    def __init__(self, dim, dim_backbone, return_layer, num_layers):
        super().__init__()
        self.dim = dim
        self.dim_backbone = dim_backbone
        self.return_layer = return_layer

        in_channel_list = [
            int(dim_backbone * 2 ** i)
            for i in range(return_layer + 1, 1)
        ]
        self.fpn    = FeaturePyramidNetwork(in_channel_list, dim)
        self.layers = nn.Sequential(
            Permute([0, 2, 3, 1]),
            SwinTransformer(dim, num_layers)
        )

    def forward(self, x):
        pyramid = OrderedDict(
            (f"{i}", x[i].tensors)
            for i in range(self.return_layer, 0)
        )
        mask = x[self.return_layer].mask
        x = self.fpn(pyramid)[f"{self.return_layer}"]
        x = self.layers(x)
        return x, mask


def inverse_sigmoid(x, eps=1e-5):
    x  = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class headhoi(nn.Module):
    """CJ"""

    def __init__(self,
        detector: nn.Module,
        body_pose_estimater: BodyPoseEstimator,
        postprocessor: nn.Module,
        feature_head: nn.Module,
        ho_matcher: nn.Module,
        triplet_decoder: nn.Module,
        num_verbs: int,
        repr_size: int = 384,
        human_idx: int = 0,
        alpha: float = 0.5,
        gamma: float = .1,
        box_score_thresh: float = .05,
        min_instances: int = 3,
        max_instances: int = 15,
        raw_lambda: float = 2.8,
    ) -> None:
        super().__init__()

        self.detector      = detector
        self.body_detector = body_pose_estimater
        self.postprocessor = postprocessor

        self.ho_matcher   = ho_matcher
        self.feature_head = feature_head
        self.kv_pe        = PositionEmbeddingSine(128, 20, normalize=True)
        self.decoder      = triplet_decoder
        self.binary_classifier = nn.Linear(repr_size, num_verbs)

        self.repr_size        = repr_size
        self.human_idx        = human_idx
        self.num_verbs        = num_verbs
        self.alpha            = alpha
        self.gamma            = gamma
        self.box_score_thresh = box_score_thresh
        self.min_instances    = min_instances
        self.max_instances    = max_instances
        self.raw_lambda       = raw_lambda

    def freeze_detector(self):
        for p in self.detector.parameters():
            p.requires_grad = False

    def compute_classification_loss(self, logits, prior, labels):
        prior = torch.cat(prior, dim=0).prod(1)
        x, y  = torch.nonzero(prior).unbind(1)

        logits = logits[:, x, y]
        prior  = prior[x, y]
        labels = labels[None, x, y].repeat(len(logits), 1)

        n_p = labels.sum()
        if dist.is_initialized():
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()

        loss = binary_focal_loss_with_logits(
            torch.log(
                prior / (1 + torch.exp(-logits) - prior) + 1e-8
            ), labels, reduction='sum',
            alpha=self.alpha, gamma=self.gamma
        )
        return loss / n_p

    def postprocessing(self, boxes, paired_inds, object_types,
                       logits, prior, image_sizes):
        n      = [len(p_inds) for p_inds in paired_inds]
        logits = logits.split(n)

        detections = []
        for bx, p_inds, objs, lg, pr, size in zip(
            boxes, paired_inds, object_types,
            logits, prior, image_sizes
        ):
            pr   = pr.prod(1)
            x, y = torch.nonzero(pr).unbind(1)
            scores = lg[x, y].sigmoid() * pr[x, y].pow(self.raw_lambda)
            detections.append(dict(
                boxes=bx, pairing=p_inds[x], scores=scores,
                labels=y, objects=objs[x], size=size, x=x
            ))
        return detections

    @staticmethod
    def base_forward(ctx, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = ctx.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        hs = ctx.transformer(ctx.input_proj(src), mask, ctx.query_embed.weight, pos[-1])[0]

        outputs_class = ctx.class_embed(hs)
        outputs_coord = ctx.bbox_embed(hs).sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out, hs, features

    def forward(self,
        images: List[Tensor],
        targets: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Parameters:
        -----------
        images: List[Tensor]
            Input images in format (C, H, W)
        targets: List[dict], optional
            Human-object interaction targets

        Returns:
        --------
        results: List[dict]
            Detected human-object interactions. Each dict has the following keys:
            `boxes`: torch.Tensor
                (N, 4) Bounding boxes for detected human and object instances
            `pairing`: torch.Tensor
                (M, 2) Pairing indices, with human instance preceding the object instance
            `scores`: torch.Tensor
                (M,) Interaction score for each pair
            `labels`: torch.Tensor
                (M,) Predicted action class for each pair
            `objects`: torch.Tensor
                (M,) Predicted object class for each pair
            `size`: torch.Tensor
                (2,) Image height and width
            `x`: torch.Tensor
                (M,) Index tensor corresponding to the duplications of human-object pairs.
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        image_sizes = torch.as_tensor(
            [im.size()[-2:] for im in images], device=images[0].device
        )

        with torch.no_grad():
            results, hs, features = self.base_forward(self.detector, images)
            results = self.postprocessor(results, image_sizes)

        region_props = prepare_region_proposals(
            results, hs[-1], image_sizes,
            box_score_thresh=self.box_score_thresh,
            human_idx=self.human_idx,
            min_instances=self.min_instances,
            max_instances=self.max_instances
        )
        boxes        = [r['boxes'] for r in region_props]
        region_props = self.body_detector(region_props, images)

        # Produce human-object pairs.
        (
            ho_queries,
            paired_inds, prior_scores,
            object_types, positional_embeds
        ) = self.ho_matcher(region_props, image_sizes)

        # Compute keys/values for triplet decoder.
        memory, mask = self.feature_head(features)
        b, h, w, c   = memory.shape
        memory       = memory.reshape(b, h * w, c)
        kv_p_m       = mask.reshape(-1, 1, h * w)
        k_pos        = self.kv_pe(NestedTensor(memory, mask)).permute(0, 2, 3, 1).reshape(b, h * w, 1, c)

        # Enhance visual context with triplet decoder.
        query_embeds = []
        for i, (ho_q, mem) in enumerate(zip(ho_queries, memory)):
            # Bug6修复：跳过空 pair，decoder 无法处理 0 长度序列
            if ho_q.shape[0] == 0:
                query_embeds.append(torch.zeros(
                    self.decoder.num_layers, 0, self.repr_size, device=ho_q.device
                ))
                continue
            query_embeds.append(self.decoder(
                ho_q.unsqueeze(1),          # (n, 1, q_dim)
                mem.unsqueeze(1),           # (hw, 1, kv_dim)
                kv_padding_mask=kv_p_m[i], # (1, hw)
                q_pos=positional_embeds[i], # centre: (n,1,512), box: (n,1,1024)
                k_pos=k_pos[i]             # (hw, 1, kv_dim)
            ).squeeze(dim=2))

        # Concatenate queries from all images in the same batch.
        query_embeds = torch.cat(query_embeds, dim=1)   # (ndec, Σn, q_dim)
        logits       = self.binary_classifier(query_embeds)

        if self.training:
            labels   = associate_with_ground_truth(boxes, paired_inds, targets, self.num_verbs)
            cls_loss = self.compute_classification_loss(logits, prior_scores, labels)
            return dict(cls_loss=cls_loss)

        detections = self.postprocessing(
            boxes, paired_inds, object_types,
            logits[-1], prior_scores, image_sizes
        )
        return detections


def build_detector(args, obj_to_verb):

    detr, _, postprocessors = build_base_detr(args)

    if os.path.exists(args.pretrained):
        if dist.is_initialized():
            print(f"Rank {dist.get_rank()}: Load weights for the object detector from {args.pretrained}")
        else:
            print(f"Load weights for the object detector from {args.pretrained}")
        detr.load_state_dict(
            torch.load(args.pretrained, map_location='cpu')['model_state_dict']
        )

    ho_matcher = HumanObjectMatcher(
        repr_size=args.repr_dim,
        num_verbs=args.num_verbs,
        obj_to_verb=obj_to_verb,
        dropout=args.dropout
    )
    decoder_layer = TransformerDecoderLayer(
        q_dim=args.repr_dim, kv_dim=args.hidden_dim,
        ffn_interm_dim=args.repr_dim * 4,
        num_heads=args.nheads, dropout=args.dropout
    )
    triplet_decoder = TransformerDecoder(
        decoder_layer=decoder_layer,
        num_layers=args.triplet_dec_layers
    )
    return_layer = {"C5": -1, "C4": -2, "C3": -3}[args.kv_src]
    if isinstance(detr.backbone.num_channels, list):
        num_channels = detr.backbone.num_channels[-1]
    else:
        num_channels = detr.backbone.num_channels

    feature_head = FeatureHead(
        args.hidden_dim, num_channels,
        return_layer, args.triplet_enc_layers
    )
    kpt_estimator       = KeypointEstimator()
    body_pose_estimator = BodyPoseEstimator(pose_model=kpt_estimator, human_idx=0)

    model = headhoi(
        detr, body_pose_estimator, postprocessors['bbox'],
        feature_head=feature_head,
        ho_matcher=ho_matcher,
        triplet_decoder=triplet_decoder,
        num_verbs=args.num_verbs,
        repr_size=args.repr_dim,
        alpha=args.alpha,
        gamma=args.gamma,
        box_score_thresh=args.box_score_thresh,
        min_instances=args.min_instances,
        max_instances=args.max_instances,
        raw_lambda=args.raw_lambda,
    )
    return model