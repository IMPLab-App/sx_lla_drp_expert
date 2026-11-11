with open("eval_confusion_matrix.py", "r", encoding="utf-8") as f:
    text = f.read()

replacement = """    args = build_args()
    
    # 填充默认参数
    from cmd_args import get_args
    default_args_parser = get_args()
    default_args = default_args_parser.parse_args([])
    for k, v in vars(default_args).items():
        if not hasattr(args, k):
            setattr(args, k, v)
            
    if not hasattr(args, 'distributed'):
        args.distributed = False
    if not hasattr(args, 'resume'):
        args.resume = False"""
        
text = text.replace("""    args = build_args()
    if not hasattr(args, 'distributed'):
        args.distributed = False
    if not hasattr(args, 'resume'):
        args.resume = False""", replacement)

with open("eval_confusion_matrix.py", "w", encoding="utf-8") as f:
    f.write(text)
