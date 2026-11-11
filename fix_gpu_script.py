with open("eval_confusion_matrix.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("    torch.cuda.set_device(args.gpu)\n    random.seed(args.seed)", "    torch.cuda.set_device(args.gpu)\n    if not use_cuda:\n        args.gpu = None\n    random.seed(args.seed)")

with open("eval_confusion_matrix.py", "w", encoding="utf-8") as f:
    f.write(text)
