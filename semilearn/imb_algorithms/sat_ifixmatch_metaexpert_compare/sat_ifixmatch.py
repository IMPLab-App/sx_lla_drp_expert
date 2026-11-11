from __future__ import annotations
import torch
import torch.nn.functional as F

def bidirectional_kl(p,q,eps=1e-8):
    p=p.clamp_min(eps); q=q.clamp_min(eps)
    return (p*(p.log()-q.log())).sum(1)+(q*(q.log()-p.log())).sum(1)

@torch.no_grad()
def detect_ood_by_prediction_consistency(pw,ps,tau_ood=0.5):
    pood=torch.clamp((pw.argmax(1)-ps.argmax(1)).abs().float(),max=1.0)
    return pood>tau_ood,pood

class SelfAdaptiveThreshold:
    def __init__(self,num_classes,lambda_t=0.5,momentum=0.9,device=None):
        self.num_classes=num_classes; self.lambda_t=lambda_t; self.momentum=momentum
        self.local=torch.full((num_classes,),1.0/num_classes,device=device)
        self.global_value=torch.tensor(1.0/num_classes,device=device)
        self.initialized=False
    @torch.no_grad()
    def update(self,probs_weak,id_mask=None):
        device=probs_weak.device
        self.local=self.local.to(device); self.global_value=self.global_value.to(device)
        if id_mask is None: id_mask=torch.ones(probs_weak.size(0),dtype=torch.bool,device=device)
        if id_mask.sum()==0: return self.thresholds()
        probs=probs_weak[id_mask]; conf,pred=probs.max(1); base=torch.tensor(1.0/self.num_classes,device=device)
        g=self.lambda_t*base+(1-self.lambda_t)*conf.mean()
        loc=torch.empty(self.num_classes,device=device)
        for k in range(self.num_classes):
            m=pred==k; loc[k]=self.lambda_t*base+(1-self.lambda_t)*(conf[m].mean() if m.any() else base)
        if not self.initialized:
            self.global_value=g; self.local=loc; self.initialized=True
        else:
            self.global_value=self.momentum*self.global_value+(1-self.momentum)*g
            self.local=self.momentum*self.local+(1-self.momentum)*loc
        return self.thresholds()
    @torch.no_grad()
    def thresholds(self):
        return self.local/self.local.max().clamp_min(1e-8)*self.global_value

def ood_uniform_loss(logits,ood_mask,temperature=100.0):
    if ood_mask.sum()==0: return logits.new_tensor(0.0)
    logp=F.log_softmax(logits[ood_mask]/temperature,dim=1)
    target=torch.full_like(logp,1.0/logp.shape[1])
    return F.kl_div(logp,target,reduction='batchmean')

def pseudo_label_loss(logits_s,probs_w,thresholds,id_mask):
    conf,pseudo=probs_w.max(1)
    selected=id_mask & (conf>thresholds[pseudo])
    if selected.sum()==0: return logits_s.new_tensor(0.0),selected,pseudo
    return F.cross_entropy(logits_s[selected],pseudo[selected]),selected,pseudo

def consistency_regularization_loss(probs_w,probs_s,id_mask):
    kl=bidirectional_kl(probs_w,probs_s)
    val=torch.where(id_mask,kl,-torch.tanh(kl))
    return val.mean()

def sat_ifixmatch_step(model,x_lb,y_lb,x_ulb,weak_aug_fn,strong_aug_fn,sat,lambda_p=1.0,lambda_o=0.5,lambda_r=0.1,tau_ood=0.5,ood_temperature=100.0):
    logits_lb=model(weak_aug_fn(x_lb)); loss_s=F.cross_entropy(logits_lb,y_lb)
    logits_w=model(weak_aug_fn(x_ulb)); logits_s=model(strong_aug_fn(x_ulb))
    probs_w=F.softmax(logits_w.detach(),1); probs_s_det=F.softmax(logits_s.detach(),1)
    ood_mask,_=detect_ood_by_prediction_consistency(probs_w,probs_s_det,tau_ood); id_mask=~ood_mask
    thr=sat.update(probs_w,id_mask=id_mask)
    loss_p,selected,pseudo=pseudo_label_loss(logits_s,probs_w,thr,id_mask)
    loss_o=ood_uniform_loss(logits_s,ood_mask,ood_temperature)
    loss_r=consistency_regularization_loss(probs_w.detach(),F.softmax(logits_s,1),id_mask)
    total=loss_s+lambda_p*loss_p+lambda_o*loss_o+lambda_r*loss_r
    stats={'loss':float(total.detach().cpu()),'loss_s':float(loss_s.detach().cpu()),'loss_p':float(loss_p.detach().cpu()),'loss_o':float(loss_o.detach().cpu()),'loss_r':float(loss_r.detach().cpu()),'selected_ratio':float(selected.float().mean().detach().cpu()),'ood_like_ratio':float(ood_mask.float().mean().detach().cpu()),'thresholds':thr.detach().cpu().tolist()}
    return total,stats

@torch.no_grad()
def predict_with_sat_rejection(model,x,weak_aug_fn,strong_aug_fn,sat,tau_ood=0.5,reject_unselected=True):
    lw=model(weak_aug_fn(x)); ls=model(strong_aug_fn(x)); pw=F.softmax(lw,1); ps=F.softmax(ls,1)
    ood_mask,_=detect_ood_by_prediction_consistency(pw,ps,tau_ood)
    conf,pred=pw.max(1); thr=sat.thresholds().to(x.device)
    reject=ood_mask | ((conf<=thr[pred]) if reject_unselected else torch.zeros_like(ood_mask))
    out=pred.clone(); out[reject]=-1
    return out,conf,thr
