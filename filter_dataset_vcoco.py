import os
import argparse
from pathlib import Path
import torch.multiprocessing as mp
from ultralytics import YOLO
from tqdm import tqdm

def worker(gpu_id, image_paths, conf_thresh, batch_size, temp_output_file):
    print(f"[GPU {gpu_id}] 启动！分配到 {len(image_paths)} 张图片。")
    model = YOLO('yolov8n.pt')
    valid_images = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc=f"GPU {gpu_id} 进度", position=gpu_id, leave=True):
        batch = [str(p) for p in image_paths[i:i+batch_size]]
        results = model(batch, device=gpu_id, conf=conf_thresh, verbose=False, stream=True)
        
        for path_str, res in zip(batch, results):
            if len(res.boxes) > 0:
                valid_images.append(Path(path_str).name)

    with open(temp_output_file, 'w', encoding='utf-8') as f:
        for name in valid_images:
            f.write(f"{name}\n")
            
    print(f"[GPU {gpu_id}] 处理完成，找到 {len(valid_images)} 张符合条件的图片。")


def main():
    parser = argparse.ArgumentParser(description="V-COCO Dual-GPU Dataset Filter")
    parser.add_argument('--data-root', default='./vcoco', type=str, help="V-COCO 数据集根目录")
    # 你的图片存在 train2014 或 val2014 里
    parser.add_argument('--partition', default='train2014', type=str, help="COCO 图片分区文件夹名")
    parser.add_argument('--batch-size', default=256, type=int, help="单张显卡的 Batch Size")
    args = parser.parse_args()

    # ==========================================
    # 核心修改：完全匹配你的截图目录结构
    # 路径： ./vcoco/mscoco2014/train2014
    # ==========================================
    image_dir = os.path.join(args.data_root, 'mscoco2014', args.partition)
    
    # 输出的 txt 文件保存在 vcoco 根目录下
    final_output_file = os.path.join(args.data_root, f'filtered_human_images_{args.partition}.txt')

    print(f"目标图片文件夹: {image_dir}")
    
    if not os.path.exists(image_dir):
        print(f"错误: 找不到图片目录 {image_dir}。请检查路径是否正确。")
        return

    image_paths = list(Path(image_dir).glob('*.*'))
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = [p for p in image_paths if p.suffix.lower() in valid_extensions]
    
    total_images = len(image_paths)
    if total_images == 0:
        print(f"错误: 在 {image_dir} 中没有找到图片。")
        return
        
    print(f"总计扫描到 {total_images} 张图片，准备开启双卡并行处理...")

    mid_point = total_images // 2
    chunks = [image_paths[:mid_point], image_paths[mid_point:]]
    temp_files = [f"temp_vcoco_gpu0.txt", f"temp_vcoco_gpu1.txt"]

    mp.set_start_method('spawn', force=True)
    processes = []
    for gpu_id in range(2):
        p = mp.Process(
            target=worker, 
            args=(gpu_id, chunks[gpu_id], 0.8, args.batch_size, temp_files[gpu_id])
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n合并双卡处理结果...")
    all_valid_names = []
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            with open(temp_file, 'r', encoding='utf-8') as f:
                all_valid_names.extend([line.strip() for line in f])
            os.remove(temp_file)

    with open(final_output_file, 'w', encoding='utf-8') as f:
        for name in all_valid_names:
            f.write(f"{name}\n")
            
    print("="*50)
    print("🚀 V-COCO 双卡清洗完毕！")
    print(f"最终保留了 {len(all_valid_names)} 张包含人体的图片。")
    print(f"过滤名单已保存至: {final_output_file}")
    print("="*50)

if __name__ == "__main__":
    main()