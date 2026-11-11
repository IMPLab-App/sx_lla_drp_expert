"""Validate prepared signal ablations before launching long training jobs."""

import ast
import importlib.util
import json
import logging
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch


batch = Path(sys.argv[1]).resolve()
manifest = json.loads((batch / "manifest.json").read_text())
code = Path(manifest["code_snapshot"])
sys.path.insert(0, str(code))
os.chdir(code)
torch.set_num_threads(4)

from semilearn.imb_algorithms.metaexpert.metaexpert import MetaExpertNet, SegFormerVectorMFF
from semilearn.nets.vit.vit_metaexpert import vit_tiny_metaexpert

# Compare the default new branch against the original code, with only the
# explicitly requested residual connection removed from the reference.
reference = (Path(manifest["source_baseline"]) / "code/semilearn/imb_algorithms/metaexpert/metaexpert.py").read_text()
reference = reference.replace("return self.out_act(fused + self.shortcut(features[-1]))", "return self.out_act(fused)")
tree = ast.parse(reference)
classes = ast.Module(body=[node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in ("SegFormerVectorMFF", "MetaExpertNet")], type_ignores=[])
namespace = {"torch": torch, "nn": torch.nn, "np": np, "F": torch.nn.functional}
exec(compile(classes, "reference_no_residual", "exec"), namespace)


def build(cls):
    torch.manual_seed(123)
    return cls(vit_tiny_metaexpert(num_classes=9), 9, torch.full((9,), 1 / 9), 0.0, 4.0, 1, 1).eval()


original_model = build(namespace["MetaExpertNet"])
model = build(MetaExpertNet)
assert all(torch.equal(v, original_model.state_dict()[k]) for k, v in model.state_dict().items())
x = torch.randn(3, 1, 1024)
with torch.no_grad():
    expected, actual = original_model(x), model(x)
    for key in ("logits", "aux_logits2", "w1", "w3", "fuse_logit_l"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def fail_if_called(module, inputs):
    raise AssertionError("A disabled module executed")


# The no-residual MFF must never execute its unused shortcut module.
hook = model.mff.shortcut.register_forward_pre_hook(fail_if_called)
with torch.no_grad():
    model(x)
hook.remove()
model.ablation_no_mff = True
hook = model.mff.register_forward_pre_hook(fail_if_called)
with torch.no_grad():
    model(x)
hook.remove()
model.ablation_no_mff = False
model.ablation_fixed_fusion = True
hooks = [module.register_forward_pre_hook(fail_if_called) for module in (model.mff, model.predict)]
with torch.no_grad():
    result = model(x)
    assert torch.equal(result["w1"], torch.full((3, 1), 0.5))
    assert torch.equal(result["w3"], torch.full((3, 1), 0.5))
    from eval_confusion_matrix_new import get_eval_logits
    torch.testing.assert_close(get_eval_logits(result), (result["logits"] + result["aux_logits2"]) / 2, rtol=0, atol=0)
for hook in hooks:
    hook.remove()
print("PASS: default branch unchanged, MFF has no residual path, disabled modules bypassed, evaluation averages experts exactly", flush=True)
del original_model, model

# Exercise the real train_step beyond fusion warmup, including gradients.
from train import get_config
from semilearn.core.utils import get_net_builder
from semilearn.imb_algorithms import get_imb_algorithm

for variant in manifest["variants"]:
    task = next(t for t in manifest["tasks"] if t["variant"] == variant and t["case"] == 1)
    sys.argv = ["verify_signal_ablation", "--c", task["config"]]
    args = get_config()
    args.distributed = False
    args.gpu = 0
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    algorithm = get_imb_algorithm(args, get_net_builder(args.net, args.net_from_name), None, logging.getLogger("verification"))
    algorithm.model = algorithm.model.cuda().train()
    algorithm.epoch = 25
    algorithm.current_epoch = 25
    algorithm.it = 0
    labels = torch.tensor([0, 1, 4, 8], device="cuda")
    signal = torch.randn(4, 1, 1024, device="cuda")
    result, log = algorithm.train_step(signal, labels, labels, signal, signal + 0.01 * torch.randn_like(signal), labels)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    for classifier in (algorithm.model.backbone.classifier, algorithm.model.backbone.aux_classifier2):
        assert classifier.weight.grad is not None and classifier.weight.grad.abs().sum() > 0
    mff_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in algorithm.model.mff.parameters())
    routing_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in algorithm.model.predict.parameters())
    assert mff_grad == (variant == "full")
    assert routing_grad == (variant != "fixed_fusion")
    if variant == "fixed_fusion":
        assert log["train/total_fuse_loss"] == 0
    else:
        assert log["train/total_fuse_loss"] > 0
    print(f"PASS: {variant} real train_step/backward; expert gradients valid; fusion loss={log['train/total_fuse_loss']:.6f}", flush=True)
    del algorithm, result, signal
    torch.cuda.empty_cache()
