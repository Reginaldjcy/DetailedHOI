"""
姿态估计模块
包含人脸姿态估计和身体姿态估计两个类
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from mmpose.apis import init_model, inference_topdown
from mmengine.registry import init_default_scope

class KeypointEstimator:
    """
    人体关键点估计器 (仅做Keypoint Detection，不包含人体检测)
    
    输入: 
        - images: torch.Tensor [B, 3, H, W] (RGB) 或 List[Tensor]
        - bboxes: 人体边界框列表
    输出: 
        - torch.Tensor [num_box, 17, 3] 包含17个COCO关键点及置信度
    """

    def __init__(
        self,
        pose_config: str = 'rtmpose-l_8xb256-420e_coco-256x192.py',
        pose_checkpoint: str = 'checkpoints/rtmpose-l_simcc-coco_pt-aic-coco_420e-256x192-1352a4d2_20230127.pth',
        device: str = None,
        kpt_conf_thres: float = 0.3
    ):
        # 初始化默认作用域（只需要调用一次）
        init_default_scope('mmpose')
        
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.kpt_conf_thres = kpt_conf_thres

        # 初始化模型
        self._init_pose_model(pose_config, pose_checkpoint)
        
    def _init_pose_model(self, config: str, checkpoint: str):
        """初始化RTMPose关键点估计器"""
        print(f"Loading RTMPose model on {self.device}...")
        try:
            self.pose_model = init_model(config, checkpoint, device=self.device)
            print("✓ RTMPose loaded")
        except Exception as e:
            raise RuntimeError(f"Failed to load RTMPose: {e}") from e

    def _prepare_image(self, image: torch.Tensor) -> np.ndarray:
        """
        将单个图像转换为MMPose所需的格式
        Input: Tensor (RGB) [C, H, W]
        Output: np.ndarray (BGR, uint8) [H, W, C]
        """
        image = image.detach()
        if image.max() <= 1.0:
            image *= 255.0
        
        # [C, H, W] -> [H, W, C]
        img_np = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        return img_bgr

    @torch.no_grad()
    def infer(
        self, 
        image: torch.Tensor, 
        bboxes: torch.Tensor
    ) -> torch.Tensor:
        """
        对给定的bboxes进行关键点估计
        
        Args:
            image: torch.Tensor [C, H, W] (RGB格式)
            bboxes: torch.Tensor [N, 4] (x1, y1, x2, y2)格式的人体边界框
            
        Returns:
            keypoints: torch.Tensor [N, 17, 3]
                      每个关键点包含 (x, y, confidence)
                      如果某个bbox没有检测到人，对应位置全为0
            
        COCO 17关键点顺序:
        0-鼻子, 1-左眼, 2-右眼, 3-左耳, 4-右耳,
        5-左肩, 6-右肩, 7-左肘, 8-右肘, 9-左腕, 10-右腕,
        11-左髋, 12-右髋, 13-左膝, 14-右膝, 15-左踝, 16-右踝
        """
        # 转换为numpy BGR格式
        img_bgr = self._prepare_image(image)
        
        # 转换bboxes为numpy格式
        if len(bboxes) == 0:
            return torch.zeros(0, 17, 3, device=bboxes.device)
        
        bboxes_np = bboxes.cpu().numpy().astype(np.float32)
        bboxes_np = np.ascontiguousarray(bboxes_np)
        
        # 使用RTMPose进行关键点估计
        pose_results = inference_topdown(self.pose_model, img_bgr, bboxes_np)
        
        # 提取关键点数据
        if len(pose_results) == 0:
            return torch.zeros(len(bboxes), 17, 3, device=bboxes.device)
        
        data_sample = pose_results[0]
        pred_instances = data_sample.pred_instances
        
        if pred_instances is None or len(pred_instances) == 0:
            return torch.zeros(len(bboxes), 17, 3, device=bboxes.device)
        
        # 获取关键点和置信度
        keypoints = pred_instances.keypoints  # (N, 17, 2)
        
        # 获取置信度
        if hasattr(pred_instances, 'keypoint_scores'):
            scores = pred_instances.keypoint_scores  # (N, 17)
        elif hasattr(pred_instances, 'keypoints_scores'):
            scores = pred_instances.keypoints_scores
        else:
            scores = np.ones((keypoints.shape[0], 17))
        
        # 组合为 [N, 17, 3] 格式 (x, y, confidence)
        keypoints_with_scores = np.concatenate(
            [keypoints, scores[:, :, None]], 
            axis=2
        )
        
        return torch.from_numpy(keypoints_with_scores).float().to(bboxes.device)


class BodyPoseEstimator:
    """
    身体姿态估计器包装类
    处理批量图像和region proposals，返回关键点
    """
    def __init__(self, pose_model, human_idx=0):
        self.pose_model = pose_model
        self.human_idx = human_idx
 

    def __call__(self, region_props, images):
            batch_size = len(images)
            
            HEAD_INDICES = [0, 1, 2, 3, 4]
            LEFT_WRIST, RIGHT_WRIST = 9, 10
            LEFT_ANKLE, RIGHT_ANKLE = 15, 16
            BODY_CENTER_INDICES = [5, 6, 11, 12]

            for i in range(batch_size):
                image = images[i]
                boxes = region_props[i]['boxes']
                labels = region_props[i]['labels']

                num_boxes = len(boxes)
                device = image.device

                keypoints   = torch.zeros(num_boxes, 17, 3, device=device)
                body_kpts_6 = torch.zeros(num_boxes, 6,  3, device=device)

                human_indices = (labels == self.human_idx).nonzero(as_tuple=True)[0]

                if len(human_indices) > 0:
                    human_boxes = boxes[human_indices]
                    with torch.no_grad():
                        pred_keypoints = self.pose_model.infer(image, human_boxes)  # [nh, 17, 3]

                    # ── 以 pred_keypoints 实际返回数量为准，防止越界 ──
                    num_valid = min(len(human_indices), len(pred_keypoints))

                    for j in range(num_valid):
                        hid  = human_indices[j]
                        kpts = pred_keypoints[j]   # [17, 3]

                        keypoints[hid] = kpts

                        # 0: 头部 (平均)
                        head_kpts = kpts[HEAD_INDICES]
                        valid = head_kpts[:, 2] > 0
                        if valid.any():
                            body_kpts_6[hid, 0, :2] = head_kpts[valid, :2].mean(0)
                            body_kpts_6[hid, 0,  2] = head_kpts[valid,  2].mean()

                        # 1: 左手腕  2: 右手腕
                        body_kpts_6[hid, 1] = kpts[LEFT_WRIST]
                        body_kpts_6[hid, 2] = kpts[RIGHT_WRIST]

                        # 3: 左脚踝  4: 右脚踝
                        body_kpts_6[hid, 3] = kpts[LEFT_ANKLE]
                        body_kpts_6[hid, 4] = kpts[RIGHT_ANKLE]

                        # 5: 身体中心 (肩膀+髋部平均)
                        center_kpts = kpts[BODY_CENTER_INDICES]
                        valid = center_kpts[:, 2] > 0
                        if valid.any():
                            body_kpts_6[hid, 5, :2] = center_kpts[valid, :2].mean(0)
                            body_kpts_6[hid, 5,  2] = center_kpts[valid,  2].mean()

                region_props[i]['keypoints']      = keypoints    # [N, 17, 3]
                region_props[i]['body_keypoints'] = body_kpts_6  # [N, 6,  3]

            return region_props