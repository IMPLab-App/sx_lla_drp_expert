"""Export actual Full-model routing weights and test probabilities, without training."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_full_prediction_data(output_dir, device='cpu'):
    import torch
    # Load the exact, independently verified implementation that produced these
    # checkpoints. The working-tree model may have a different architecture.
    analysis = ROOT / 'analysis/cola_two_kl'
    sys.path.insert(0, str(analysis))
    from frozen_diagnostics import runtime_model_types
    from analyze_results import data_splits

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(2)
    vit, wrapper = runtime_model_types()
    data_file = ROOT / 'data/raw_signal_dataset_G_1024.npz'
    with np.load(data_file, allow_pickle=False) as data:
        raw = data['data'].astype(np.float32)
        all_labels = data['labels']
        names = data['class_names']
    splits = data_splits(all_labels, seed=2)
    mean = float(raw[splits['train']].mean(dtype=np.float64))
    std = max(float(raw[splits['train']].std(dtype=np.float64)), 1e-8)
    test = torch.from_numpy((raw[splits['test']] - mean) / std).unsqueeze(1)
    labels = all_labels[splits['test']]
    manifest = json.loads((ROOT / 'config/cola_two_kl_24/manifest.json').read_text())
    paths, provenance = [], []
    for task in manifest['tasks']:
        if task['variant'] != 'full':
            continue
        run = int(task['run_name'].split('_')[1])
        checkpoint_path = ROOT / task['model_dir'] / 'model_best.pth'
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state = checkpoint['ema_model']
        model = wrapper(vit.vit_tiny_metaexpert(num_classes=9), 9, state['hat'],
                        float(state['tau1']), float(state['tau3']), 1, 1).to(device).eval()
        model.load_state_dict(state, strict=True)
        raw_weights, final_weights, logits = [], [], []
        with torch.inference_mode():
            for offset in range(0, len(test), 64):
                outputs = model(test[offset:offset + 64].to(device))
                raw_weights.append(outputs['fuse_w_logit'].softmax(1).cpu().numpy())
                final_weights.append(torch.cat((outputs['w1'], outputs['w3']), 1).cpu().numpy())
                logits.append((outputs['w1'] * outputs['logits'] + outputs['w3'] * outputs['aux_logits2']).cpu())
        routing = np.concatenate(raw_weights)
        final = np.concatenate(final_weights)
        fused = torch.cat(logits)
        probabilities = fused.softmax(1).numpy()
        # Match the previously verified historical predictions, not reference-art values.
        cached_path = analysis / f'predictions_full_{run}.npz'
        with np.load(cached_path, allow_pickle=False) as cached:
            assert np.array_equal(labels, cached['labels'])
            assert np.allclose(final[:, 1:2], cached['w3'], atol=2e-5)
            assert np.allclose(fused.numpy(), cached['fused'], atol=3e-4, rtol=2e-5)
            assert np.array_equal(fused.argmax(1).numpy(), cached['fused'].argmax(1))
        assert np.allclose(routing.sum(1), 1, atol=1e-6)
        assert np.allclose(final.sum(1), 1, atol=1e-6)
        assert np.allclose(probabilities.sum(1), 1, atol=1e-6)
        path = output_dir / f'full_run_{run}_plot_data.npz'
        np.savez_compressed(path, run=run, labels=labels, class_names=names,
                            dea_weights=routing, final_weights=final, probabilities=probabilities)
        paths.append(path)
        provenance.append(dict(run=run, run_name=task['run_name'], checkpoint=str(checkpoint_path),
                               checkpoint_sha256=sha256(checkpoint_path), source_predictions=str(cached_path),
                               source_predictions_sha256=sha256(cached_path), exported_file=str(path.resolve()),
                               accuracy=float(np.mean(probabilities.argmax(1) == labels)), samples=len(labels),
                               routing_range=[float(routing.min()), float(routing.max())]))
        print(f'Full run {run}: {len(labels)} actual predictions verified; accuracy={provenance[-1]["accuracy"]:.4%}', flush=True)
        del model, checkpoint, state
    (output_dir / 'provenance.json').write_text(json.dumps(dict(
        dataset=str(data_file), data_sha256=sha256(data_file), seed=2,
        runtime_source=str(analysis / 'runtime_snapshot_verified'),
        normalization=dict(mean=mean, std=std),
        dea_weights='softmax(fuse_w_logit), the raw Head/Tail routing probabilities',
        final_weights='model output w1 / w3 after confidence post-processing',
        records=provenance), ensure_ascii=False, indent=2) + '\n')
    return paths


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'results/publication_figures/full/data')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    export_full_prediction_data(args.output_dir, args.device)
