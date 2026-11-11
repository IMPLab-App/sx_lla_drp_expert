with open("eval_confusion_matrix.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("balanced_dset = BasicDataset(balanced_data, balanced_targets, args.num_classes, False", 
                    "balanced_dset = BasicDataset(balanced_data, balanced_targets, None, args.num_classes, False")
text = text.replace("lt_dset = BasicDataset(lt_data, lt_targets, args.num_classes, False", 
                    "lt_dset = BasicDataset(lt_data, lt_targets, None, args.num_classes, False")

with open("eval_confusion_matrix.py", "w", encoding="utf-8") as f:
    f.write(text)
