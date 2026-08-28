import os
import shutil
import argparse
from pathlib import Path
import torch.multiprocessing as mp
from tqdm import tqdm

def copy_worker(gpu_id, task_list, src_dir, dst_dir):
    """
    虽然拷贝图片主要靠磁盘 I/O，但开多进程可以并行文件操作
    """
    print(f"[Worker {gpu_id}] 开始搬运 {len(task_list)} 张图片...")
    for img_name in tqdm(task_list, desc=f"Worker {gpu_id}", position=gpu_id):
        src_path = os.path.join(src_dir, img_name)
        dst_path = os.path.join(dst_dir, img_name)
        
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path) # copy2 会保留元数据
        else:
            print(f"找不到原图: {src_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract filtered images to a new folder")
    parser.add_argument('--data-root', default='./hicodet', type=str)
    parser.add_argument('--partition', default='train2015', type=str)
    # 原始图片路径 (根据你的 HICO-DET 实际存放位置调整)
    parser.add_argument('--src-sub-dir', default='hico_20160224_det/images', type=str)
    args = parser.parse_args()

    # 1. 路径定义
    src_dir = os.path.join(args.data_root, args.src_sub_dir, args.partition)
    dst_dir = os.path.join(args.data_root, 'filtered_images', args.partition)
    filter_file = os.path.join(args.data_root, f'filtered_human_images_{args.partition}.txt')

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    # 2. 读取名单
    if not os.path.exists(filter_file):
        print(f"错误: 找不到名单 {filter_file}")
        return
    
    with open(filter_file, 'r') as f:
        img_names = [line.strip() for line in f]

    print(f"准备从 {src_dir} 提取 {len(img_names)} 张图片到 {dst_dir}")

    # 3. 多进程分工 (虽然不用 GPU，但多进程能榨干 CPU 和磁盘性能)
    num_processes = 8  # 搬运工可以多开点
    chunk_size = len(img_names) // num_processes
    processes = []

    for i in range(num_processes):
        start = i * chunk_size
        end = None if i == num_processes - 1 else (i + 1) * chunk_size
        p = mp.Process(target=copy_worker, args=(i, img_names[start:end], src_dir, dst_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n✅ 数据提取完成！新数据集位置: " + dst_dir)

if __name__ == "__main__":
    main()