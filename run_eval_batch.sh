#!/bin/bash
for i in {2..8}; do
    prefix=$(printf "%03d" $i)
    # Find the corresponding saved_models folder
    model_dir=$(ls -d saved_models/${prefix}*)
    config_file=$(ls config/_linux_ratio_sweep/${prefix}*_linux_eval.yaml)
    
    echo "============================================="
    echo "Evaluating model $model_dir with config $config_file"
    python eval_confusion_matrix.py --c $config_file --load_path ${model_dir}/model_best.pth
done
