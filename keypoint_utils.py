"""
keypoint_utils.py
骨架插值工具函数 —— 独立文件，避免循环导入
detailhoi1.py 和 build_model_pose1.py 都从这里导入
"""
import torch

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

# 每条连线插入的中间点数（不含端点）
# 改这里即可调整密度，其余代码自动适配
NUM_INTERP_PER_EDGE = 1

# 总关键点数 = 17(原始) + 14(连线数) × NUM_INTERP_PER_EDGE
TOTAL_KEYPOINTS = 17 + len(SKELETON_CONNECTIONS) * NUM_INTERP_PER_EDGE  # 默认 31


def compute_iou(a, b):
    """a, b 均为 [x1, y1, x2, y2]"""
    x1 = max(a[0], b[0]);  y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]);  y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def attach_body_keypoints_to_region_props(region_props, body_result, human_idx=0):
    """
    在原始17个COCO关键点基础上，沿骨架连线插值生成中间点。

    Args:
        region_props : list[dict]  每张图的 proposal 字典
        body_result  : dict[int -> list[body_dict]]
            body_dict: {"bbox": [x1,y1,x2,y2], "conf": float,
                        "keypoints": [[x,y,score], ...]}  长度17
        human_idx    : int  人类的标签索引

    Writes:
        region_props[i]['body_keypoints']: Tensor (n, TOTAL_KEYPOINTS, 3)
            前17个 = 原始COCO关键点
            后续   = 骨架连线插值点，conf = min(端点1_conf, 端点2_conf)
    """
    for i, rp in enumerate(region_props):
        boxes  = rp["boxes"]
        labels = rp["labels"]
        device = boxes.device
        n      = len(boxes)

        body_kpts = torch.zeros((n, TOTAL_KEYPOINTS, 3), device=device)
        human_ids = (labels == human_idx).nonzero(as_tuple=True)[0]

        if i not in body_result or len(body_result[i]) == 0:
            rp["body_keypoints"] = body_kpts
            continue

        bodies = body_result[i]

        for hid in human_ids:
            hb = boxes[hid].cpu().tolist()
            best_body, best_iou = None, 0.0

            for b in bodies:
                iou = compute_iou(hb, b["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_body = b

            if best_body is None:
                continue

            kpts_full = torch.tensor(best_body["keypoints"], device=device)  # (17, 3)

            # 前17个：原始关键点
            body_kpts[hid, :17] = kpts_full

            # 后续：骨架连线插值点
            idx = 17
            for (j1, j2) in SKELETON_CONNECTIONS:
                p1 = kpts_full[j1]
                p2 = kpts_full[j2]
                for k in range(1, NUM_INTERP_PER_EDGE + 1):
                    t           = k / (NUM_INTERP_PER_EDGE + 1)
                    interp_xy   = (1 - t) * p1[:2] + t * p2[:2]
                    interp_conf = torch.min(p1[2], p2[2])
                    body_kpts[hid, idx] = torch.cat(
                        [interp_xy, interp_conf.unsqueeze(0)]
                    )
                    idx += 1

        rp["body_keypoints"] = body_kpts