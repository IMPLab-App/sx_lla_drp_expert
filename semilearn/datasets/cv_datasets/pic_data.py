import os
import random
import math
import numpy as np
from PIL import Image
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.utils import split_labeled_unlabeled_data
from semilearn.datasets.augmentation import RandAugment

# 使用 ImageNet 的均值和方差作为默认值，适用于大多数自然图像
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def load_custom_data_from_folder(root_dir, img_size):
    """
    遍历文件夹读取数据
    Structure: root_dir/class_name/image_file
    """
    data = []
    targets = []
    
    # 获取类别列表并排序，保证索引对应关系固定
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    print(f"检测到类别: {class_to_idx}")

    for cls_name in classes:
        cls_dir = os.path.join(root_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
                
            img_path = os.path.join(cls_dir, img_name)
            try:
                # 使用 PIL 读取并转为 RGB，防止读取到 RGBA 或 灰度图报错
                img = Image.open(img_path).convert('RGB')
                # Resize 到统一大小，如果是工业缺陷图建议不要太小，但要为了打包成 numpy 数组
                # 这里为了和 CIFAR 逻辑一致，先 resize 再转 array
                img = img.resize((img_size, img_size)) 
                img_np = np.asarray(img)
                
                data.append(img_np)
                targets.append(class_to_idx[cls_name])
            except Exception as e:
                print(f"Error reading {img_path}: {e}")

    return np.array(data, dtype=np.uint8), np.array(targets)

def get_pic(args, name, data_dir='./data', include_lb_to_ulb=False):
    """
    适配自定义数据的加载函数
    """
    # 假设你的数据放在 data_dir/pic_data 下
    # 例如: ./data/pic_data/滑靴磨粒磨损/xxx.jpg
    root_path = os.path.join(data_dir, 'pic_data') 
    
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"Data directory not found: {root_path}")

    # 1. 加载所有数据到内存
    all_data, all_targets = load_custom_data_from_folder(root_path, args.img_size)
    num_total = len(all_targets)
    
    if num_total == 0:
        raise ValueError("No images found in the directory.")

    # 2. 手动划分 Train (用于 Labeled/Unlabeled) 和 Test (Eval)
    # 这一步是因为 CIFAR 自带 split，但自定义数据通常混在一起
    indices = np.arange(num_total)
    np.random.shuffle(indices)
    
    test_ratio = 0.2  # 20% 作为测试集，可根据需要调整
    split_idx = int(num_total * (1 - test_ratio))
    
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    train_data = all_data[train_idx]
    train_targets = all_targets[train_idx]
    test_data = all_data[test_idx]
    test_targets = all_targets[test_idx]

    # 3. 定义数据增强 (Transforms) - 保持原 CIFAR 代码逻辑
    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_medium = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # 4. 使用 Semilearn 的工具划分有标签和无标签数据
    lb_idx, ulb_idx, lb_clean_idx, lb_noise_idx = split_labeled_unlabeled_data(args, train_data, train_targets,
                                                                               num_classes=args.num_classes,
                                                                               lb_num_labels=args.num_labels,
                                                                               ulb_num_labels=args.ulb_num_labels,
                                                                               lb_imbalance_ratio=args.lb_imb_ratio,
                                                                               ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                               noise_ratio=args.noise_ratio,
                                                                               noise_per_class=args.noise_per_class,
                                                                               lb_imb_type=args.lb_imb_type,
                                                                               ulb_imb_type=args.ulb_imb_type,
                                                                               num_steps=args.num_steps,
                                                                               include_lb_to_ulb=include_lb_to_ulb)

    # 5. 处理标签噪音逻辑 (与原代码保持一致)
    data, targets = train_data, train_targets
    noised_targets = targets.copy() # 初始化
    
    lb_count = [0 for _ in range(args.num_classes)]
    lb_clean_count = [0 for _ in range(args.num_classes)]
    lb_noise_count = [0 for _ in range(args.num_classes)]
    new_lb_noise_count = [0 for _ in range(args.num_classes)]
    ulb_count = [0 for _ in range(args.num_classes)]

    for c in targets[lb_idx]:
        lb_count[c] += 1
    for c in targets[ulb_idx]:
        ulb_count[c] += 1
    for c in targets[lb_clean_idx]:
        lb_clean_count[c] += 1
    if len(lb_noise_idx) > 0:
        for c in targets[lb_noise_idx]:
            lb_noise_count[c] += 1

    # 计算噪音概率矩阵 (仅当 noise_ratio > 0 时生效)
    p_noise = np.zeros((args.num_classes, args.num_classes))
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            if i != j:
                # 防止除以0
                denominator = (sum(lb_count) - lb_count[i])
                if denominator > 0:
                    p_noise[i][j] = lb_count[j] / denominator
    
    row_sums = p_noise.sum(axis=1, keepdims=True)
    # 处理全0行防止 NaN
    row_sums[row_sums == 0] = 1 
    p_noise = p_noise / row_sums

    for i in lb_noise_idx:
        if args.noise_type == 'sym':
            noised_targets[i] = (random.randint(1, args.num_classes - 1) + targets[i]) % args.num_classes
        elif args.noise_type == 'asym':
            noised_targets[i] = np.random.choice(args.num_classes, p=p_noise[targets[i]])
        elif args.noise_type == 'circle':
            noised_targets[i] = (targets[i] + 1) % args.num_classes
    if len(lb_noise_idx) > 0:
        for c in noised_targets[lb_noise_idx]:
            new_lb_noise_count[c] += 1

    print(f"Data Loaded: Total {num_total} images.")
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    # 6. 构建 Dataset 对象
    lb_dset = BasicDataset(data[lb_idx], targets[lb_idx], noised_targets[lb_idx], args.num_classes, False,
                           weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    ulb_dset = BasicDataset(data[ulb_idx], targets[ulb_idx], None, args.num_classes, True,
                            weak_transform=transform_weak, strong_transform=transform_strong, onehot=False)

    eval_dset = BasicDataset(test_data, test_targets, None, args.num_classes, False, weak_transform=transform_val,
                             strong_transform=None, onehot=False)

    lb_count_message = {'lb_count': lb_count, 'ulb_count': ulb_count, 'lb_clean_count': lb_clean_count,
                        'lb_noise_count': lb_noise_count, 'new_lb_noise_count': new_lb_noise_count}

    return lb_dset, ulb_dset, eval_dset, lb_count_message