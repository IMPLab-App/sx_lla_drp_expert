from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

def parse_list(s,typ=float): return [] if s is None or s.strip()=='' else [typ(v.strip()) for v in s.split(',') if v.strip()!='']
def exp_rates(k,ir): return [1.0] if k<=1 else [(1.0/ir)**(c/(k-1)) for c in range(k)]

def main():
    p=argparse.ArgumentParser(); p.add_argument('--full_npz',required=True); p.add_argument('--out_dir',required=True); p.add_argument('--num_classes',type=int,required=True); p.add_argument('--ood_classes',default=''); p.add_argument('--max_per_class',type=int,default=None); p.add_argument('--lt_rates',default=None); p.add_argument('--exp_imbalance_ratio',type=float,default=None); p.add_argument('--label_ratio',type=float,default=0.1); p.add_argument('--test_ratio',type=float,default=0.2); p.add_argument('--seed',type=int,default=0); args=p.parse_args()
    rng=np.random.default_rng(args.seed); a=np.load(args.full_npz); x=a['x']; y=a['y'].astype(np.int64); out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); ood=set(parse_list(args.ood_classes,int))
    if args.lt_rates: rates=parse_list(args.lt_rates,float); assert len(rates)==args.num_classes
    elif args.exp_imbalance_ratio is not None: rates=exp_rates(args.num_classes,args.exp_imbalance_ratio)
    else: rates=[1.0]*args.num_classes
    lb=[]; ulb=[]; test=[]
    for c in range(args.num_classes):
        idx=np.where(y==c)[0]; rng.shuffle(idx)
        if args.max_per_class is not None:
            n=min(max(1,int(round(args.max_per_class*rates[c]))),len(idx)); idx=idx[:n]
        nt=max(1,int(round(len(idx)*args.test_ratio))) if len(idx)>=3 else 0; test_c=idx[:nt]; train_c=idx[nt:]
        nl=max(1,int(round(len(train_c)*args.label_ratio))) if len(train_c)>0 else 0; lb+=train_c[:nl].tolist(); ulb+=train_c[nl:].tolist(); test+=test_c.tolist()
    for c in ood:
        idx=np.where(y==c)[0]; rng.shuffle(idx); nt=max(1,int(round(len(idx)*args.test_ratio))) if len(idx)>=3 else 0; test+=idx[:nt].tolist(); ulb+=idx[nt:].tolist()
    rng.shuffle(lb); rng.shuffle(ulb); rng.shuffle(test); yulb=y[ulb].copy(); ytest=y[test].copy()
    for c in ood: yulb[yulb==c]=-1; ytest[ytest==c]=-1
    np.savez(out/'train_lb.npz',x=x[lb],y=y[lb]); np.savez(out/'train_ulb.npz',x=x[ulb],y=yulb); np.savez(out/'test.npz',x=x[test],y=ytest)
    meta={'num_classes':args.num_classes,'ood_classes':sorted(list(ood)),'rates':rates,'n_lb':len(lb),'n_ulb':len(ulb),'n_test':len(test),'lb_counts_known':{str(c):int((y[lb]==c).sum()) for c in range(args.num_classes)},'ulb_counts_known_plus_ood':{str(c):int((yulb==c).sum()) for c in list(range(args.num_classes))+[-1]},'test_counts_known_plus_ood':{str(c):int((ytest==c).sum()) for c in list(range(args.num_classes))+[-1]}}
    (out/'split_meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(meta,ensure_ascii=False,indent=2)); print(f'Saved splits to {out}')
if __name__=='__main__': main()
