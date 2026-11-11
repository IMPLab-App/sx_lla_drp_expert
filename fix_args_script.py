with open("eval_confusion_matrix.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("global args\n    args = build_args()", "global args\n    args = build_args()\n    args.distributed = False\n    args.resume = False")
text = text.replace("    args = build_args()", "    args = build_args()\n    if not hasattr(args, 'distributed'):\n        args.distributed = False\n    if not hasattr(args, 'resume'):\n        args.resume = False")

with open("eval_confusion_matrix.py", "w", encoding="utf-8") as f:
    f.write(text)
