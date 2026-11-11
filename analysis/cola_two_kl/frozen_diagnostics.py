"""Post-hoc inference from isolated runtime code; never modifies saved models.

All official heldout confusion matrices must reproduce before diagnostics are
accepted. The unlabeled analysis uses unaugmented, normalized train samples;
it is not a reconstruction of historical weak-augmentation predictions.
"""
import argparse
import ast
import csv
import importlib.util
import json
import os
from pathlib import Path
from functools import partial
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support

sys.path.insert(0,str(Path(__file__).resolve().parent))
from analyze_results import ROOT, OUT, SNAPSHOT, data_splits, numerical_module, write_csv


def runtime_model_types():
    spec=importlib.util.spec_from_file_location('experiment_vit',SNAPSHOT/'semilearn/nets/vit/vit_metaexpert.py')
    vit=importlib.util.module_from_spec(spec);spec.loader.exec_module(vit)
    source=(SNAPSHOT/'semilearn/imb_algorithms/metaexpert/metaexpert.py').read_text()
    tree=ast.parse(source)
    nodes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name in ('SegFormerVectorMFF','MetaExpertNet')]
    namespace=dict(torch=torch,nn=nn,np=np,F=F,normalize_distribution=numerical_module().normalize_distribution)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(SNAPSHOT/'metaexpert_classes.py'),'exec'),namespace)
    return vit,namespace['MetaExpertNet']


def collect(model,values,device,batch_size=64,counter_tau=None):
    output={key:[] for key in ('head','tail','fused','features','w3')}
    if counter_tau is not None: output['counter_fused']=[]
    saved_tau=float(model.tau3)
    with torch.inference_mode():
        for a in range(0,len(values),batch_size):
            x=values[a:a+batch_size].to(device)
            result=model(x)
            output['head'].append(result['logits'].cpu().numpy())
            output['tail'].append(result['aux_logits2'].cpu().numpy())
            output['fused'].append((result['w1']*result['logits']+result['w3']*result['aux_logits2']).cpu().numpy())
            output['features'].append(result['feat'].cpu().numpy())
            output['w3'].append(result['w3'].cpu().numpy())
            if counter_tau is not None:
                model.tau3.fill_(counter_tau)
                changed=model(x)
                output['counter_fused'].append((changed['w1']*changed['logits']+changed['w3']*changed['aux_logits2']).cpu().numpy())
                model.tau3.fill_(saved_tau)
    return {key:np.concatenate(value) for key,value in output.items()}


def main(device='cpu'):
    torch.set_num_threads(4);torch.manual_seed(2)
    vit,wrapper=runtime_model_types();ddde=numerical_module()
    with np.load(ROOT/'data/raw_signal_dataset_G_1024.npz',allow_pickle=False) as source:
        raw=source['data'].astype(np.float32);labels=source['labels']
    splits=data_splits(labels)
    mean=float(raw[splits['train']].mean(dtype=np.float64));std=float(raw[splits['train']].std(dtype=np.float64))
    normalized=(torch.from_numpy(raw)-mean)/max(std,1e-8)
    normalized=normalized.unsqueeze(1)
    test_x=normalized[splits['test']];test_y=labels[splits['test']]
    train_x=normalized[splits['train']];train_y=labels[splits['train']]
    tasks=json.loads((ROOT/'config/cola_two_kl_24/manifest.json').read_text())['tasks']
    metadata=json.loads((OUT/'data_metadata.json').read_text())['runs']
    branch_rows=[];confidence_rows=[];estimator_rows=[];geometry_rows=[];counter_rows=[];paired=[];predictions={}
    for variant in ('full','no_ddde_kl'):
        for task in [t for t in tasks if t['variant']==variant]:
            run=int(task['run_name'].split('_')[1]);folder=ROOT/task['model_dir']
            state=torch.load(folder/'model_best.pth',map_location='cpu',weights_only=False)
            net=wrapper(vit.vit_tiny_metaexpert(num_classes=9),9,state['ema_model']['hat'],
                float(state['ema_model']['tau1']),float(state['ema_model']['tau3']),1,1).to(device).eval()
            net.load_state_dict(state['ema_model'],strict=True)
            result=collect(net,test_x,device,counter_tau=4. if variant=='full' else None)
            pred=result['fused'].argmax(1)
            counts=confusion_matrix(test_y,pred,labels=list(range(9)))
            official=json.loads((folder/'evaluation/evaluation.json').read_text())
            expected=np.rint(np.array(official['heldout']['confusion_matrix'])*100).astype(int)
            if not np.array_equal(counts,expected):
                raise RuntimeError(f'Historical predictions did not reproduce: {variant}/{run}; delta={counts-expected}')
            predictions[(variant,run)]=pred
            np.savez_compressed(OUT/f'predictions_{variant}_{run}.npz',labels=test_y,head=result['head'],tail=result['tail'],
                fused=result['fused'],w3=result['w3'],**({'counter_fused':result['counter_fused']} if variant=='full' else {}))
            for branch in ('head','tail','fused'):
                p=result[branch].argmax(1)
                recalls=np.array([np.mean(p[test_y==k]==k) for k in range(9)])
                branch_rows.append(dict(variant=variant,run=run,branch=branch,accuracy=float(np.mean(p==test_y)),
                    macro_f1=float(f1_score(test_y,p,average='macro')),head_accuracy=float(recalls[:3].mean()),
                    middle_accuracy=float(recalls[3:6].mean()),tail_accuracy=float(recalls[6:].mean()),
                    **{f'recall_c{k}':float(recalls[k]) for k in range(9)}))
            z=result['features'];z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
            centers=np.stack([z[test_y==k].mean(0) for k in range(9)])
            within=np.array([np.square(z[test_y==k]-centers[k]).sum(1).mean() for k in range(9)])
            for a,b in [(3,4),(3,6),(4,6)]:
                separation=float(np.square(centers[a]-centers[b]).sum())
                geometry_rows.append(dict(variant=variant,run=run,class_a=a,class_b=b,
                    centroid_distance_squared=separation,within_scatter=float(within[a]+within[b]),
                    fisher_ratio=separation/max(float(within[a]+within[b]),1e-12)))
            if variant=='full':
                altered=result['counter_fused'].argmax(1)
                counter_rows.append(dict(run=run,best_tau=float(net.tau3),official_accuracy=float(np.mean(pred==test_y)),
                    frozen_model_tau4_accuracy=float(np.mean(altered==test_y)),
                    official_macro_f1=float(f1_score(test_y,pred,average='macro')),
                    frozen_model_tau4_macro_f1=float(f1_score(test_y,altered,average='macro')),
                    changed_predictions=int(np.sum(pred!=altered))))
                indices=np.load(metadata[str(run)]['unlabeled_indices']);uy=train_y[indices]
                unlabeled=collect(net,train_x[indices],device)
                tail=torch.from_numpy(unlabeled['tail']);features=torch.from_numpy(unlabeled['features'])
                probs,guess=tail.softmax(1).max(1);selected=probs>=.95
                q=np.bincount(uy,minlength=9)/len(uy)
                hist_all=np.bincount(guess.numpy(),minlength=9)/len(guess)
                selected_count=np.bincount(guess[selected].numpy(),minlength=9)
                hist_selected=selected_count/max(selected_count.sum(),1)
                rng=np.random.RandomState(2)
                rank_distributions=[]
                for step in range(64):
                    sampled=torch.from_numpy(rng.randint(0,len(uy),128))
                    mask=selected[sampled];chosen=guess[sampled][mask];feats=features[sampled][mask]
                    ranks=torch.stack([ddde.effective_rank(feats[chosen==k]) for k in range(9)])
                    if ranks.sum()>0: rank_distributions.append((ranks/ranks.sum()).numpy())
                rank_mean=np.mean(rank_distributions,axis=0)
                for k in range(9):
                    actual=uy==k;retain=selected.numpy()[actual]
                    chosen_mask=(guess.numpy()==k)&selected.numpy()
                    confidence_rows.append(dict(run=run,class_id=k,true_samples=int(actual.sum()),
                        selection_coverage=float(retain.mean()),raw_tail_recall=float(np.mean(guess.numpy()[actual]==k)),
                        selected_prediction_precision=float(np.mean(uy[chosen_mask]==k)) if chosen_mask.any() else '',
                        true_prior=float(q[k]),predicted_prior=float(hist_all[k]),selected_prior=float(hist_selected[k]),
                        simulated_rank_prior=float(rank_mean[k])))
                estimator_rows.append(dict(run=run,raw_pseudo_accuracy=float(np.mean(guess.numpy()==uy)),
                    selected_fraction=float(selected.float().mean()),
                    selected_accuracy=float(np.mean(guess.numpy()[selected.numpy()]==uy[selected.numpy()])),
                    tv_prediction_hist=float(np.abs(hist_all-q).sum()/2),
                    tv_selected_hist=float(np.abs(hist_selected-q).sum()/2),
                    tv_rank64_batch_mean=float(np.abs(rank_mean-q).sum()/2),simulated_batches=len(rank_distributions)))
            print(f'PASS reproduced official heldout confusion matrix: {variant} run {run}; accuracy={np.mean(pred==test_y):.6f}',flush=True)
            del net,state,result
            if str(device).startswith('cuda'):torch.cuda.empty_cache()
    for run in range(1,7):
        a=predictions[('full',run)]==test_y;b=predictions[('no_ddde_kl',run)]==test_y
        full_only=int(np.sum(a&~b));fixed_only=int(np.sum(~a&b))
        discordant=full_only+fixed_only
        paired.append(dict(run=run,full_only_correct=full_only,fixed_only_correct=fixed_only,
            net_correct_difference=full_only-fixed_only,exact_mcnemar_p=float(binomtest(full_only,discordant,.5).pvalue) if discordant else 1.))
    write_csv('branch_diagnostics.csv',branch_rows);write_csv('confidence_diagnostics.csv',confidence_rows)
    write_csv('estimator_posthoc.csv',estimator_rows);write_csv('feature_geometry.csv',geometry_rows)
    write_csv('frozen_tau_counterfactual.csv',counter_rows);write_csv('paired_test_diagnostics.csv',paired)
    (OUT/'inference_provenance.json').write_text(json.dumps(dict(device=str(device),threads=4,models=12,official_confusion_matrices_reproduced=12,
        runtime_source=str(SNAPSHOT),dataset='raw_signal_dataset_G_1024.npz',seed=2,mean=mean,std=std,
        note='Post-hoc diagnostics. No model updates or historical-result replacement. Unlabeled subset has no augmentation.'),indent=2))
    print('COUNTERFACTUAL',json.dumps(counter_rows),flush=True)
    print('ESTIMATOR_POSTHOC',json.dumps(estimator_rows),flush=True)
    print('PAIRED',json.dumps(paired),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu')
    main(parser.parse_args().device)
