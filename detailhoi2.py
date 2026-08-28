"""
CJ
ho_q:                            
q_pos: skeleton (31) / object - n1:n2 - n1+n2=512
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

from build_model_pose1 import BodyPoseEstimator, KeypointEstimator

from keypoint_utils import (
    attach_body_keypoints_to_region_props,
    TOTAL_KEYPOINTS,
    SKELETON_CONNECTIONS,
    NUM_INTERP_PER_EDGE,
)

# ── 修改这两个值来控制 centre_feat 的两段维度，总和必须等于 512 ──
HUMAN_CENTRE_DIM = 256   # human skeleton PE 输出维度
OBJECT_CENTRE_DIM = 256  # object centre PE 输出维度


# ---------------------------------------------------------
# 骨架连线配置（COCO 17关键点索引）
# 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
# 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow
# 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip
# 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
# ---------------------------------------------------------
SKELETON_CONNECTIONS = [
    # 躯干
    (5,  6),   # 左肩 - 右肩
    (5,  11),  # 左肩 - 左髋
    (6,  12),  # 右肩 - 右髋
    (11, 12),  # 左髋 - 右髋
    # 左臂
    (5,  7),   # 左肩 - 左肘
    (7,  9),   # 左肘 - 左手腕
    # 右臂
    (6,  8),   # 右肩 - 右肘
    (8,  10),  # 右肘 - 右手腕
    # 左腿
    (11, 13),  # 左髋 - 左膝
    (13, 15),  # 左膝 - 左脚踝
    # 右腿
    (12, 14),  # 右髋 - 右膝
    (14, 16),  # 右膝 - 右脚踝
    # 头部到躯干
    (0,  5),   # 鼻子 - 左肩
    (0,  6),   # 鼻子 - 右肩
]

# 每条连线上插入的中间点数量（不含端点）
NUM_INTERP_PER_EDGE = 1

# 总关键点数 = 17(原始) + 14(连线数) × NUM_INTERP_PER_EDGE
TOTAL_KEYPOINTS = 17 + len(SKELETON_CONNECTIONS) * NUM_INTERP_PER_EDGE  # 默认31


# ---------------------------------------------------------
# Utility: IOU
# ---------------------------------------------------------
def compute_iou(a, b):
    """ a,b are [x1,y1,x2,y2] """
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------
# attach body keypoints into region_props
# 原始17个关键点 + 骨架连线插值点
# ---------------------------------------------------------
def attach_body_keypoints_to_region_props(region_props, body_result, human_idx=0):
    """
    region_props: list of dict of each image
    body_result: dict[int -> list[body dict]]
        body dict: {"bbox": [x1,y1,x2,y2], "conf": float, "keypoints": [[x,y,score], ...]}

    Adds:
        rp["body_keypoints"]: (n, TOTAL_KEYPOINTS, 3)
            - 前17个: 原始COCO关键点 [x, y, score]
            - 后续:   骨架连线插值点 [x, y, score]
                      插值点的 score = min(端点1_score, 端点2_score)

    若修改插值密度，只需调整顶部的 NUM_INTERP_PER_EDGE 常量。
    """
    for i, rp in enumerate(region_props):
        boxes  = rp["boxes"]
        labels = rp["labels"]
        device = boxes.device
        n = len(boxes)

        # 初始化: (n, TOTAL_KEYPOINTS, 3)
        body_kpts = torch.zeros((n, TOTAL_KEYPOINTS, 3), device=device)

        human_ids = (labels == human_idx).nonzero(as_tuple=True)[0]

        # 没有检测到人体
        if i not in body_result or len(body_result[i]) == 0:
            rp["body_keypoints"] = body_kpts
            continue

        bodies = body_result[i]

        # 将每个 human box 匹配到最优 body（最大IoU）
        for hid in human_ids:
            hb = boxes[hid].cpu().numpy()
            best_body, best_iou = None, 0.0

            for b in bodies:
                iou = compute_iou(hb, b["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_body = b

            if best_body is None:
                continue

            kpts_full = torch.tensor(best_body["keypoints"], device=device)  # (17, 3)

            # ---- 前17个: 原始关键点 ----
            body_kpts[hid, :17] = kpts_full

            # ---- 后续: 骨架连线插值点 ----
            idx = 17
            for (j1, j2) in SKELETON_CONNECTIONS:
                p1 = kpts_full[j1]  # [x, y, conf]
                p2 = kpts_full[j2]  # [x, y, conf]

                for k in range(1, NUM_INTERP_PER_EDGE + 1):
                    t = k / (NUM_INTERP_PER_EDGE + 1)
                    interp_xy   = (1 - t) * p1[:2] + t * p2[:2]
                    interp_conf = torch.min(p1[2], p2[2])
                    body_kpts[hid, idx] = torch.cat(
                        [interp_xy, interp_conf.unsqueeze(0)]
                    )
                    idx += 1

        rp["body_keypoints"] = body_kpts


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
    def __init__(self, repr_size, num_verbs, obj_to_verb,
                 human_dim=HUMAN_CENTRE_DIM, obj_dim=OBJECT_CENTRE_DIM,
                 dropout=.1, human_idx=0):
        super().__init__()
        self.repr_size = repr_size
        self.num_verbs = num_verbs
        self.human_idx = human_idx
        self.obj_to_verb = obj_to_verb
        self.human_centre_dim = human_dim   # human skeleton PE 输出维度
        self.object_centre_dim = obj_dim    # object centre PE 输出维度

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
        # human: TOTAL_KEYPOINTS(31) × 256 → human_centre_dim
        self.kpts_pe_proj = nn.Linear(TOTAL_KEYPOINTS * 256, self.human_centre_dim)
        # object: 256 → object_centre_dim
        self.obj_c_pe_proj = nn.Linear(256, self.object_centre_dim)

        # 对 region-level embedding 做一个浅层 encoder
        self.encoder = TransformerEncoder(num_layers=2, dropout=dropout)

        # multimodal fusion: [human,obj] visual (512) + spatial (repr_size)
        self.mmf = MultiModalFusion(512, repr_size, repr_size)

        # body keypoints MLP head: TOTAL_KEYPOINTS × 2 (x, y) → human_centre_dim
        self.body_head = nn.Sequential(
            nn.Linear(TOTAL_KEYPOINTS * 2, 128), nn.ReLU(),
            nn.Linear(128, self.human_centre_dim)
        )

    def check_human_instances(self, labels):
        is_human = labels == self.human_idx
        nh = torch.sum(is_human)
        if not torch.all(labels[:nh] == self.human_idx):
            raise AssertionError("Human instances are not permuted to the top!")
        return nh

    def compute_box_pe(self, boxes, embeds, body_kpts, image_size):
        """
        boxes     : (N, 4)
        embeds    : (N, 256)
        body_kpts : (N, TOTAL_KEYPOINTS, 3)   ← x, y, conf
        image_size: (2,)  H, W

        returns:
            box_pe:      (N, 512)
            c_pe_out:    (N, object_centre_dim)
            kpts_pe_out: (N, human_centre_dim)
        """
        bx_norm = boxes / image_size[[1, 0, 1, 0]]
        bx_c  = (bx_norm[:, :2] + bx_norm[:, 2:]) / 2
        b_wh  = bx_norm[:, 2:] - bx_norm[:, :2]

        kpts_norm = body_kpts.clone()
        kpts_norm[..., 0] /= image_size[1]  # x / W
        kpts_norm[..., 1] /= image_size[0]  # y / H
        kpts_xy = kpts_norm[..., :2]         # (N, TOTAL_KEYPOINTS, 2)

        c_pe   = compute_sinusoidal_pe(bx_c[:, None], 20).squeeze(1)    # (N, 256)
        wh_pe  = compute_sinusoidal_pe(b_wh[:, None], 20).squeeze(1)    # (N, 256)
        kpts_pe = compute_sinusoidal_pe(kpts_xy, 20)                    # (N, TOTAL_KEYPOINTS, 256)

        box_pe = torch.cat([c_pe, wh_pe], dim=-1)                       # (N, 512)

        ref_hw_cond = self.ref_anchor_head(embeds).sigmoid()
        c_pe[..., :128] *= (ref_hw_cond[:, 1] / b_wh[:, 1]).unsqueeze(-1)
        c_pe[..., 128:] *= (ref_hw_cond[:, 0] / b_wh[:, 0]).unsqueeze(-1)

        kpts_conf  = body_kpts[..., 2]
        valid_mask = kpts_conf > 0.3

        N = kpts_xy.shape[0]
        kpts_wh = torch.zeros(N, 2, device=kpts_xy.device)

        for i in range(N):
            valid_kpts_i = kpts_xy[i][valid_mask[i]]
            if len(valid_kpts_i) > 0:
                kpts_min = valid_kpts_i.min(0)[0]
                kpts_max = valid_kpts_i.max(0)[0]
                kpts_wh[i] = kpts_max - kpts_min
            else:
                kpts_wh[i] = torch.ones(2, device=kpts_xy.device)

        ref_kpts_wh = self.ref_keypoint_head(embeds).sigmoid()
        kpts_scale  = ref_kpts_wh / (kpts_wh + 1e-6)

        kpts_pe[..., :128] *= kpts_scale[:, 1].unsqueeze(-1).unsqueeze(-1)
        kpts_pe[..., 128:] *= kpts_scale[:, 0].unsqueeze(-1).unsqueeze(-1)

        # ── 投影到各自目标维度 ──
        kpts_pe_out = self.kpts_pe_proj(kpts_pe.reshape(N, -1))  # (N, human_centre_dim)
        c_pe_out    = self.obj_c_pe_proj(c_pe)                   # (N, object_centre_dim)

        return box_pe, c_pe_out, kpts_pe_out

    # ------------------------------------------------------------------------
    # MAIN FORWARD
    # ------------------------------------------------------------------------
    def forward(self, region_props, image_sizes, device=None):
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
            body_kpts = rp["body_keypoints"]   # (n, TOTAL_KEYPOINTS, 3)

            nh = self.check_human_instances(labels)
            n  = len(boxes)

            # ---------------- 枚举 HO pairs ----------------
            x, y = torch.meshgrid(
                torch.arange(n, device=device),
                torch.arange(n, device=device),
                indexing='ij'
            )
            x_keep, y_keep = torch.nonzero(
                torch.logical_and(x != y, x < nh)
            ).unbind(1)

            if len(x_keep) == 0:
                ho_queries.append(torch.zeros(0, self.repr_size, device=device))
                paired_indices.append(torch.zeros(0, 2, device=device, dtype=torch.int64))
                prior_scores.append(torch.zeros(0, 2, self.num_verbs, device=device))
                object_types.append(torch.zeros(0, device=device, dtype=torch.int64))
                positional_embeds.append({
                    "centre": torch.zeros(0, 1, self.human_centre_dim + self.object_centre_dim, device=device),
                    "box":    torch.zeros(0, 1, 1024, device=device),
                })
                continue

            # ---------------- 空间编码 ----------------
            pairwise_spatial = compute_spatial_encodings(
                [boxes[x.flatten()],], [boxes[y.flatten()],], [image_sizes[i],]
            )
            pairwise_spatial = self.spatial_head(pairwise_spatial)
            pairwise_spatial = pairwise_spatial.reshape(n, n, -1)

            # ---------------- positional encoding ----------------
            box_pe, c_pe_out, kpts_pe_out = self.compute_box_pe(
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

            # ---------------- positional embeds for decoder ----------------
            centre_feat = torch.cat([
                kpts_pe_out[x_keep],  # human keypoint PE (human_centre_dim)
                c_pe_out[y_keep],     # object center PE  (object_centre_dim)
            ], dim=-1)                # (num_pairs, human_centre_dim + object_centre_dim)

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