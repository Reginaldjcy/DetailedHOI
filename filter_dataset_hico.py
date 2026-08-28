import os
import argparse
from pathlib import Path
import torch.multiprocessing as mp
from ultralytics import YOLO
from tqdm import tqdm

def worker(gpu_id, image_paths, conf_thresh, batch_size, temp_output_file):
    """
    独立的工作进程，绑定特定的 GPU 运行
    """
    print(f"[GPU {gpu_id}] 启动！分配到 {len(image_paths)} 张图片。")
    
    # 初始化模型，指定加载到对应的显卡上
    model = YOLO('yolov8n.pt')
    
    valid_images = []
    
    # 将分配给这块卡的图片，再切分成一个个 Batch 进行处理
    # position=gpu_id 确保两个进度条不会在终端里互相覆盖
    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"GPU {gpu_id} 进度", position=gpu_id, leave=True):
        batch = [str(p) for p in image_paths[i:i+batch_size]]
        
        # 1. 传入整个 batch 列表
        # 2. device=gpu_id 确保在正确的卡上推理
        # 3. stream=True 是处理大量数据的关键，防止内存爆炸
        results = model(batch, device=gpu_id, conf=conf_thresh, verbose=False, stream=True)
        
        for path_str, res in zip(batch, results):
            if len(res.boxes) > 0:
                valid_images.append(Path(path_str).name)

    # 将当前 GPU 的结果写入专属的临时文件，避免多进程抢占写入冲突
    with open(temp_output_file, 'w', encoding='utf-8') as f:
        for name in valid_images:
            f.write(f"{name}\n")
            
    print(f"[GPU {gpu_id}] 处理完成，找到 {len(valid_images)} 张符合条件的图片。")


def main():
    parser = argparse.ArgumentParser(description="Dual-GPU Dataset Filter")
    parser.add_argument('--data-root', default='./hicodet', type=str)
    parser.add_argument('--partition', default='train2015', type=str)
    # A6000 有 48G 显存，yolov8n 非常小，Batch Size 开到 256 甚至 512 都毫无压力
    parser.add_argument('--batch-size', default=256, type=int, help="单张显卡的 Batch Size")
    args = parser.parse_args()

    image_dir = os.path.join(args.data_root, 'hico_20160224_det', 'images', args.partition)
    final_output_file = os.path.join(args.data_root, f'filtered_human_images_{args.partition}.txt')

    # 获取所有图片路径
    image_paths = list(Path(image_dir).glob('*.*'))
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = [p for p in image_paths if p.suffix.lower() in valid_extensions]
    
    total_images = len(image_paths)
    if total_images == 0:
        print(f"错误: 在 {image_dir} 中没有找到图片。")
        return
        
    print(f"总计扫描到 {total_images} 张图片，准备开启双卡并行处理...")

    # 将数据集平均切分为两份
    mid_point = total_images // 2
    chunks = [
        image_paths[:mid_point],   # 给 GPU 0 的数据
        image_paths[mid_point:]    # 给 GPU 1 的数据
    ]
    
    temp_files = [f"temp_gpu0.txt", f"temp_gpu1.txt"]

    # 启动多进程 (必须使用 spawn 模式来兼容 CUDA)
    mp.set_start_method('spawn', force=True)
    processes = []
    
    for gpu_id in range(2): # 遍历 0 和 1
        p = mp.Process(
            target=worker, 
            args=(gpu_id, chunks[gpu_id], 0.8, args.batch_size, temp_files[gpu_id])
        )
        p.start()
        processes.append(p)

    # 等待两张卡都跑完
    for p in processes:
        p.join()

    # 合并临时文件
    print("\n合并双卡处理结果...")
    all_valid_names = []
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding='utf-8') as f:
                all_valid_names.extend([line.strip() for line in f])
            os.remove(temp_file) # 删掉临时文件

    # 写入最终结果
    with open(final_output_file, 'w', encoding='utf-8') as f:
        for name in all_valid_names:
            f.write(f"{name}\n")
            
    print("="*50)
    print("🚀 双卡清洗完毕！")
    print(f"最终保留了 {len(all_valid_names)} 张图片。")
    print(f"文件已保存至: {final_output_file}")
    print("="*50)

if __name__ == "__main__":
    main()