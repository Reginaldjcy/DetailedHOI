import json
import os

# 探测原始 COCO 标注位置
# 通常在 vcoco 目录下还会有原生的 instances_train2014.json 等
possible_coco_annos = [
    'vcoco/mscoco2014/annotations/instances_train2014.json',
    'vcoco/mscoco2014/annotations/instances_val2014.json'
]

target_id = 450

for p in possible_coco_annos:
    if os.path.exists(p):
        print(f"正在检查: {p}")
        with open(p, 'r') as f:
            data = json.load(f)
        
        # 查找图片
        img = [i for i in data['images'] if i['id'] == target_id]
        if img:
            print(f"✅ 找到图片! 文件名: {img[0]['file_name']}")
            # 查找该图所有物体标注
            annos = [a for a in data['annotations'] if a['image_id'] == target_id]
            cat_map = {c['id']: c['name'] for c in data['categories']}
            
            print(f"该图共有 {len(annos)} 个标注物体:")
            for a in annos:
                print(f"- {cat_map[a['category_id']]}: {a['bbox']}")
            break
else:
    print("在常见的 MS-COCO 标注路径下也没找到 450 号图。")
    print("建议检查一下你的 vcoco 文件夹下是否还有其他 .json 文件？")