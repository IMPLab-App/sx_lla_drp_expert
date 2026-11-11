import re

with open("eval_confusion_matrix.py", "r", encoding="utf-8") as f:
    text = f.read()

idx = text.find("def main():")
before = text[:idx]

new_code = """import math
from semilearn.datasets.utils import make_imbalance_data
from semilearn.datasets.cv_datasets.datasetbase import BasicDataset

def evaluate_and_plot(net, eval_loader, save_root, args, class_names, device, use_cuda, suffix_title):
    y_true = []
    y_pred = []
    all_probs = []

    with torch.no_grad():
        for data in eval_loader:
            x = data["x_lb"]
            y = data["y_lb"]

            if isinstance(x, dict):
                x = {k: v.to(device, non_blocking=use_cuda) for k, v in x.items()}
            else:
                x = x.to(device, non_blocking=use_cuda)
            y = y.to(device, non_blocking=use_cuda)

            outputs = net(x)
            if isinstance(outputs, dict):
                logits = outputs.get("logits", None)
                if logits is None:
                    logits = next(iter(outputs.values()))
            else:
                logits = outputs

            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            y_true.extend(y.cpu().tolist())
            y_pred.extend(pred.cpu().tolist())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    all_probs = np.asarray(all_probs, dtype=np.float64)
    acc = float((y_true == y_pred).mean()) if len(y_true) else 0.0
    cf_mat = __import__("sklearn.metrics", fromlist=["confusion_matrix"]).confusion_matrix(y_true, y_pred, normalize="true")
    cf_mat = np.nan_to_num(cf_mat, nan=0.0) 

    os.makedirs(save_root, exist_ok=True)

    csv_path = os.path.join(save_root, f"model_best_{suffix_title}.csv")
    png_path = os.path.join(save_root, f"model_best_{suffix_title}.png")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + class_names)
        for name, row in zip(class_names, cf_mat):
            writer.writerow([name] + [f"{value:.6f}" for value in row])

    mean_matrix, std_matrix, class_counts = build_avg_probability_matrix(
        all_probs, y_true, args.num_classes
    )
    mean_path, std_path, mean_std_path = save_avg_probability_matrix(
        mean_matrix, std_matrix, class_names, save_root
    )
    # rename for suffix
    mean_path_new = os.path.join(save_root, f"avg_prob_mean_matrix_{suffix_title}.csv")
    std_path_new = os.path.join(save_root, f"avg_prob_std_matrix_{suffix_title}.csv")
    mean_std_path_new = os.path.join(save_root, f"avg_prob_mean_std_matrix_{suffix_title}.csv")
    os.replace(mean_path, mean_path_new)
    os.replace(std_path, std_path_new)
    os.replace(mean_std_path, mean_std_path_new)
    
    avg_prob_png_path = os.path.join(save_root, f"avg_prob_mean_std_matrix_{suffix_title}.png")
    plot_avg_probability_matrix(mean_matrix, std_matrix, class_names, avg_prob_png_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_cjk_font(plt)

    fig_size = max(6, args.num_classes * 0.8)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cf_mat, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(args.num_classes),
        yticks=np.arange(args.num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix ({suffix_title}) - Eval Accuracy: {acc:.2%}",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cf_mat.max() / 2.0 if cf_mat.size > 0 else 0.0
    for i in range(args.num_classes):
        for j in range(args.num_classes):
            ax.text(
                j,
                i,
                f"{cf_mat[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if cf_mat[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[{suffix_title}] Test Accuracy: {acc:.4f}")
    print(f"[{suffix_title}] Confusion matrix image saved to: {png_path}")
    print(f"[{suffix_title}] Average probability mean/std image saved to: {avg_prob_png_path}")

def main():
    args = build_args()

    use_cuda = args.gpu is not None and args.gpu >= 0 and torch.cuda.is_available()
    device = torch.device(f"cuda:{args.gpu}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(args.gpu)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = resolve_checkpoint_path(args.load_path)

    # 1. 完整初始化算法，这样 MetaExpert 会读取真实的所有设置
    net_builder = get_net_builder(args.net, args.net_from_name)
    if hasattr(args, 'imb_algorithm') and args.imb_algorithm is not None:
        from semilearn.imb_algorithms import get_imb_algorithm
        import logging
        logging.getLogger().setLevel(logging.ERROR)
        algorithm = get_imb_algorithm(args, net_builder, None, logging.getLogger("eval"))
    else:
        from semilearn.algorithms import get_algorithm
        import logging
        logging.getLogger().setLevel(logging.ERROR)
        algorithm = get_algorithm(args, net_builder, None, logging.getLogger("eval"))
        
    # 我们不仅构建了骨架，接下来加载 checkpoint
    try:
        algorithm.load_model(checkpoint_path)
    except Exception as e:
        print(f"[Warning] using safe load_state_dict. Error: {e}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("ema_model", checkpoint.get("model"))
        algorithm.ema_model.load_state_dict(state_dict)

    # 评测使用 ema_model
    net = algorithm.ema_model.to(device)
    net.eval()

    # Rebuild the same eval split as training uses.
    _, _, eval_dset, _ = get_raw_signal(
        args,
        name=args.dataset,
        data_dir=args.data_dir,
        include_lb_to_ulb=args.include_lb_to_ulb,
    )

    data = eval_dset.data
    targets = np.array(eval_dset.targets)
    
    # 构建平衡数据集
    # 找出最小分类的数量
    class_indices = {c: np.where(targets == c)[0] for c in range(args.num_classes)}
    # Avoid classes with 0 samples
    min_count = min(len(indices) for c, indices in class_indices.items() if len(indices) > 0)
    
    print(f"Min elements found in any class in the test set to build balanced set: {min_count}")
    
    balanced_indices = []
    for c in range(args.num_classes):
        if len(class_indices[c]) > 0:
            c_idx = np.random.choice(class_indices[c], min_count, replace=False)
            balanced_indices.extend(c_idx)
        
    balanced_indices = np.array(balanced_indices)
    balanced_data = [data[i] for i in balanced_indices] if isinstance(data, list) else data[balanced_indices]
    balanced_targets = targets[balanced_indices]
    
    balanced_dset = BasicDataset(args.dataset, balanced_data, balanced_targets, args.num_classes, False, weak_transform=eval_dset.weak_transform, strong_transform=None, onehot=False)

    balanced_loader = DataLoader(
        balanced_dset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=math.inf if hasattr(math, 'inf') else max(1, args.num_workers),
    )

    # 构建长尾数据集 (使用 lb_imb_ratio 缩放)
    num_classes = args.num_classes
    target_counts = make_imbalance_data(100000, num_classes, args.lb_imb_ratio, "exp", num_steps=0) # 这里的 100000 只是为了算出比例
    proportions = np.array(target_counts) / target_counts[0]
    
    # 为了最大化利用测试集，我们要找到符合这个比例的最大 scale
    max_scale = float('inf')
    for c in range(num_classes):
        if len(class_indices[c]) == 0 and proportions[c] > 0:
            print(f"[Warning] Class {c} has 0 samples in eval set but requires {proportions[c]} proportion. Ignoring proportion limits for this class.")
            continue
        if proportions[c] > 0:
            scale_c = len(class_indices[c]) / proportions[c]
            if scale_c < max_scale:
                max_scale = scale_c
                
    lt_indices = []
    for c in range(num_classes):
        if proportions[c] > 0 and max_scale != float('inf'):
            c_count = int(max_scale * proportions[c])
            if c_count > 0 and len(class_indices[c]) >= c_count:
                c_idx = np.random.choice(class_indices[c], c_count, replace=False)
                lt_indices.extend(c_idx)

    lt_indices = np.array(lt_indices)
    lt_data = [data[i] for i in lt_indices] if isinstance(data, list) else data[lt_indices]
    lt_targets = targets[lt_indices]
    print(f"Total imbalanced tests target length: {len(lt_targets)}")
    
    lt_dset = BasicDataset(args.dataset, lt_data, lt_targets, args.num_classes, False, weak_transform=eval_dset.weak_transform, strong_transform=None, onehot=False)
    
    lt_loader = DataLoader(
        lt_dset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    save_root = args.output_dir
    if not save_root:
        if os.path.isdir(checkpoint_path):
            save_root = checkpoint_path
        else:
            save_root = os.path.dirname(checkpoint_path)
    save_root = os.path.join(save_root, "confusion_matrix")
    
    class_names = get_pic_class_names(args.data_dir)
    if len(class_names) != args.num_classes:
        class_names = [str(i) for i in range(args.num_classes)]
    else:
        class_names = [str(x) for x in class_names]

    # Evaluate Balanced
    print("Evaluating balanced testing set:")
    evaluate_and_plot(net, balanced_loader, save_root, args, class_names, device, use_cuda, "balanced")
    
    # Evaluate Long Tail
    print("Evaluating imbalanced testing set (matching lb_imb_ratio):")
    evaluate_and_plot(net, lt_loader, save_root, args, class_names, device, use_cuda, "imbalanced")
    
if __name__ == '__main__':
    main()
"""

with open("eval_confusion_matrix_new2.py", "w", encoding="utf-8") as f:
    f.write(before + new_code)
