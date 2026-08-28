"""
CJ
"""

import os
import sys
import torch
import random
import warnings
import argparse
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from detailhoi1 import build_detector
from utils import custom_collate, CustomisedDLE, DataFactory
from configs_d import detector_args

warnings.filterwarnings("ignore")

###################
###################################################################################
from torch.utils.data import DataLoader, DistributedSampler, Subset

class FilteredDataFactory:
    """直接包装 DataFactory，用过滤后的索引重映射，保留所有原始属性"""
    def __init__(self, data_factory, valid_indices):
        # 必须用 object.__setattr__ 直接写，避免触发自定义 __getattr__
        object.__setattr__(self, '_factory', data_factory)
        object.__setattr__(self, '_indices', valid_indices)
        
    def __len__(self):
        return len(self._indices)
    
    def __getitem__(self, i):
        return self._factory[self._indices[i]]
    
    def __getattr__(self, name):
        # _factory 和 _indices 由 object.__setattr__ 写入，不会走到这里
        # 其他所有属性透传给 DataFactory
        return getattr(object.__getattribute__(self, '_factory'), name)
    
    @property
    def dataset(self):
        # 保持 .dataset 链路：FilteredDataFactory.dataset → VCOCO底层
        return object.__getattribute__(self, '_factory').dataset

def get_filtered_indices(dataset_obj, partition_name, rank, args):
    """返回过滤后的有效索引列表（基于 DataFactory 的相对索引）"""
    valid_names = set()
    
    if args.dataset == 'vcoco':
        files_to_load = [
            os.path.join(args.data_root, 'filtered_human_images_train2014.txt'),
            os.path.join(args.data_root, 'filtered_human_images_val2014.txt')
        ]
    else:
        files_to_load = [
            os.path.join(args.data_root, f'filtered_human_images_{partition_name}.txt')
        ]
    
    loaded_any = False
    for f_path in files_to_load:
        if os.path.exists(f_path):
            if rank == 0:
                print(f"=> 加载过滤名单: {f_path}")
            with open(f_path, 'r', encoding='utf-8') as f:
                valid_names.update(line.strip() for line in f)
            loaded_any = True
    
    if not loaded_any:
        if rank == 0:
            print(f"=> 未找到过滤名单，[{partition_name}] 使用完整数据集")
        return list(range(len(dataset_obj)))
    
    # ✅ 关键：用 DataFactory 的相对索引 0..len-1 来过滤
    # DataFactory.__getitem__(i) -> VCOCO.__getitem__(i) -> _keep[i]
    # 所以这里的 i 就是 DataFactory 对外暴露的合法索引
    vcoco_dataset = dataset_obj.dataset  # 底层 VCOCO 对象
    
    valid_indices = []
    for i in range(len(dataset_obj)):  # 遍历 DataFactory 的合法索引范围
        # 通过 VCOCO 的 filename(i) 获取文件名，与白名单比对
        fname = vcoco_dataset.filename(i)
        if fname in valid_names:
            valid_indices.append(i)
    
    if rank == 0:
        print(f"=> [{partition_name}] 过滤: {len(dataset_obj)} -> {len(valid_indices)} 条")
    
    return valid_indices
#####################################################


def main(rank, args):

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    # Fix seed
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.set_device(rank)


    trainset = DataFactory(
        name=args.dataset, partition=args.partitions[0],
        data_root=args.data_root
    )
    testset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root
    )

    train_indices = get_filtered_indices(trainset, args.partitions[0], rank, args)
    test_indices  = get_filtered_indices(testset,  args.partitions[1], rank, args)

    # 用 FilteredDataFactory 包装，不破坏 .dataset 链路
    trainset_filtered = FilteredDataFactory(trainset, train_indices)
    testset_filtered  = FilteredDataFactory(testset,  test_indices)

    train_loader = DataLoader(
        dataset=trainset_filtered,
        collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            trainset_filtered, num_replicas=args.world_size,
            rank=rank, drop_last=True)
    )
    test_loader = DataLoader(
        dataset=testset_filtered,
        collate_fn=custom_collate, batch_size=args.batch_size // args.world_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            testset_filtered, num_replicas=args.world_size,
            rank=rank, drop_last=True)
    )

    # .dataset 链路恢复正常，utils.py 里的 dataloader.dataset.dataset 拿到的就是 VCOCO
    if args.dataset == 'hicodet':
        object_to_target = train_loader.dataset.dataset.object_to_verb
        args.num_verbs = 117
    elif args.dataset == 'vcoco':
        object_to_target = list(train_loader.dataset.dataset.object_to_action.values())
        args.num_verbs = 24

    
    model = build_detector(args, object_to_target)

    if os.path.exists(args.resume):
        print(f"=> Rank {rank}: PViC loaded from saved checkpoint {args.resume}.")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print(f"=> Rank {rank}: PViC randomly initialised.")

    engine = CustomisedDLE(model, train_loader, test_loader, args)

    if args.cache:
        if args.dataset == 'hicodet':
            engine.cache_hico(test_loader, args.output_dir)
        elif args.dataset == 'vcoco':
            engine.cache_vcoco(test_loader, args.output_dir)
        return

    if args.eval:
        if args.dataset == 'vcoco':
            """
            NOTE This evaluation results on V-COCO do not necessarily follow the 
            protocol as the official evaluation code, and so are only used for
            diagnostic purposes.
            """
            ap = engine.test_vcoco()
            if rank == 0:
                print(f"The mAP is {ap.mean():.4f}.")
            return
        else:
            ap = engine.test_hico()
            if rank == 0:
                # Fetch indices for rare and non-rare classes
                rare = trainset.dataset.rare
                non_rare = trainset.dataset.non_rare
                print(
                    f"The mAP is {ap.mean():.4f},"
                    f" rare: {ap[rare].mean():.4f},"
                    f" none-rare: {ap[non_rare].mean():.4f}"
                )
            return

    model.freeze_detector()
    param_dicts = [{"params": [p for p in model.parameters() if p.requires_grad]}]
    optim = torch.optim.AdamW(param_dicts, lr=args.lr_head, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, args.lr_drop, gamma=args.lr_drop_factor)
    # Override optimiser and learning rate scheduler
    engine.update_state_key(optimizer=optim, lr_scheduler=lr_scheduler)

    engine(args.epochs)

@torch.no_grad()
def sanity_check(args):
    dataset = DataFactory(name='hicodet', partition=args.partitions[0], data_root=args.data_root)
    args.num_verbs = 117
    args.num_triplets = 600
    object_to_target = dataset.dataset.object_to_verb
    model = build_detector(args, object_to_target)
    if args.eval:
        model.eval()
    if os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        print(f"Loading checkpoints from {args.resume}.")
        model.load_state_dict(ckpt['model_state_dict'])

    image, target = dataset[998]
    outputs = model([image], targets=[target])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(parents=[detector_args(),])
    parser.add_argument('--raw-lambda', default=2.8, type=float)

    parser.add_argument('--kv-src', default='C5', type=str, choices=['C5', 'C4', 'C3'])
    parser.add_argument('--repr-dim', default=384, type=int)
    parser.add_argument('--triplet-enc-layers', default=1, type=int)
    parser.add_argument('--triplet-dec-layers', default=2, type=int)

    parser.add_argument('--alpha', default=.5, type=float)
    parser.add_argument('--gamma', default=.1, type=float)
    parser.add_argument('--box-score-thresh', default=.05, type=float)
    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--use-wandb', default=False, action='store_true')

    parser.add_argument('--port', default='1234', type=str)
    parser.add_argument('--seed', default=140, type=int)
    parser.add_argument('--world-size', default=2, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--sanity', action='store_true')

    args = parser.parse_args()
    print(args)

    if args.sanity:
        sanity_check(args)
        sys.exit()
    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.port

    mp.spawn(main, nprocs=args.world_size, args=(args,))
