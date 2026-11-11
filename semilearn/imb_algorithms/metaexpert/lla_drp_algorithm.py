"""Independent E3 LLA+DRP integration; active LLA-only source is unchanged."""
import copy
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from semilearn.algorithms.utils import SSL_Argument, str2bool
from semilearn.core.utils import IMB_ALGORITHMS
from .metaexpert import MetaExpertNet
from .lla_algorithm import LLAE3MetaExpert
from .lla_runtime import IsolatedUpdateHook
from .drp_e3 import DRPE3Controller


class MFFRetainedExpertNet(MetaExpertNet):
    @classmethod
    def wrap(cls, original, fixed_weights=False):
        result=cls.__new__(cls)
        result.__dict__=original.__dict__.copy()
        result._modules=original._modules.copy()
        result.fixed_dea_weights=fixed_weights
        return result

    def tail_logit_adjustment(self):
        return self.lla.adjustment(detach=True) if hasattr(self,'lla') else super().tail_logit_adjustment()

    def forward(self, x, **kwargs):
        # Always run original lin1-lin8, predict and both DEA softmax stages.
        out=super().forward(x,**kwargs)
        out['dynamic_w1'],out['dynamic_w3']=out['w1'],out['w3']
        if self.fixed_dea_weights:
            head,tail=out['logits'],out['aux_logits2']
            out['w1']=head.new_full((len(head),1),.5)
            out['w3']=head.new_full((len(head),1),.5)
            adjusted_head=head+self.tau1*torch.log(self.hat)
            adjusted_tail=tail+self.tail_logit_adjustment()
            # Preserve the existing per-expert minimum subtraction in the
            # fusion training logits. It is a per-sample classwise constant.
            positive=lambda value:value-value.min(-1,keepdim=True).values
            out['fuse_logit_l']=.5*positive(adjusted_head)+.5*positive(adjusted_tail)
            out['fuse_logit_HMT_c_1']=.5*positive(out['logitsH'])+.5*positive(out['logitsT'])
            out['fuse_logit_HMT_c_3']=.5*positive(out['aux_logitsH2'])+.5*positive(out['aux_logitsT2'])
            # fuse_w_logit remains the original trainable predict output.
        return out


class DRPUpdateHook(IsolatedUpdateHook):
    def after_train_step(self,a):
        a._fusion_gradient_squares={'mff':[],'predict':[]}
        # Optimizer pre-step hooks observe gradients after AMP unscale, clipping
        # and native DP/DDP reduction.
        a._gradient_scale_used=1.
        super().after_train_step(a)


@IMB_ALGORITHMS.register('metaexpert_lla_drp')
class LLADRPMetaExpert(LLAE3MetaExpert):
    def __init__(self,args,net_builder,tb_log=None,logger=None):
        if args.ablation_no_dea_aggregator or args.ablation_no_mff:
            raise ValueError('This study always retains MFF; legacy removal switches are forbidden')
        if args.e3_drp_enabled and args.e3_adjust_mode!='lla_hard':
            raise ValueError('Primary DRP study requires the LLA-adjusted E3 weak prediction')
        if args.ablation_fixed_dea_weights and not (args.e3_drp_enabled and args.e3_adjust_mode=='lla_hard'):
            raise ValueError('M2 requires both LLA and DRP')
        self.drp_enabled=args.e3_drp_enabled
        self.fixed_dea_weights=args.ablation_fixed_dea_weights
        self._gradient_handles=[]
        self._fusion_gradient_squares={'mff':[],'predict':[]}
        self._gradient_scale_used=1.
        self._mff_active_steps=0
        self._mff_nonzero_steps=0
        self._drp_last=None
        self._dea_last=None
        super().__init__(args,net_builder,tb_log,logger)
        for name in ('model','ema_model'):
            setattr(self,name,MFFRetainedExpertNet.wrap(getattr(self,name),self.fixed_dea_weights))
        self.drp=(DRPE3Controller(self.num_classes,args.drp_momentum,args.drp_n_sigma,
            args.drp_per_class,args.drp_epsilon).to(self.p_hat_lb.device) if self.drp_enabled else None)
        self._drp_before_step=None

    def set_hooks(self):
        super().set_hooks()
        for index,hook in enumerate(self._hooks):
            if hook.name=='ParamUpdateHook':
                new=DRPUpdateHook();new.name,new.priority=hook.name,hook.priority
                self._hooks[index]=new;self.hooks_dict[hook.name]=new

    def finalize_model_setup(self):
        super().finalize_model_setup()
        if self.drp is not None:self.drp.to(next(self.model.parameters()).device)
        for handle in self._gradient_handles:handle.remove()
        self._gradient_handles=[]
        def capture_optimizer_gradients(optimizer,arguments,keywords):
            self._fusion_gradient_squares={'mff':[],'predict':[]}
            for name,parameter in self.module.named_parameters():
                group='mff' if any(name.startswith(f'lin{i}.') for i in range(1,9)) else 'predict' if name.startswith('predict.') else None
                if group is not None and parameter.grad is not None:
                    self._fusion_gradient_squares[group].append(parameter.grad.detach().float().square().sum())
        self._gradient_handles.append(self.optimizer.register_step_pre_hook(capture_optimizer_gradients))

    def _lla_expert_loss(self,outputs,num_lb,labels):
        if not self.drp_enabled:
            return super()._lla_expert_loss(outputs,num_lb,labels)
        logits=outputs['aux_logits2']
        weak,_=logits[num_lb:].chunk(2)
        _,strong_h=outputs['aux_logitsH2'][num_lb:].chunk(2)
        _,strong_t=outputs['aux_logitsT2'][num_lb:].chunk(2)
        supervised=self.ce_loss(logits[:num_lb],labels,reduction='mean')
        with torch.no_grad(),torch.autocast(device_type=logits.device.type,enabled=False):
            probabilities=self.module.lla.adjust(weak.detach()).softmax(-1)
            # Identical weak probabilities feed hard pseudo labels and DRP.
            weights,confidence,gap=self.drp(probabilities)
            pseudo=probabilities.argmax(-1)
            effective=weights*(confidence>=self.p_cutoff).float() if self.args.drp_with_hard_mask else weights
            head_weight=effective.detach()
            tail_weight=head_weight*(pseudo>=self.cut1).float()
        loss_h=self.ce_loss(strong_h,pseudo,reduction='none')
        loss_t=self.ce_loss(strong_t,pseudo,reduction='none')
        unsupervised=((loss_h.float()*head_weight).sum()+(loss_t.float()*tail_weight).sum())/(head_weight.sum()+tail_weight.sum()+self.args.drp_loss_epsilon)
        if not torch.isfinite(unsupervised):raise FloatingPointError('Non-finite DRP E3 H/T loss')
        accepted=effective>0
        counts=torch.bincount(pseudo[accepted],minlength=self.num_classes)
        conf_sum=confidence.new_zeros(self.num_classes).scatter_add_(0,pseudo[accepted],confidence[accepted])
        self._pending_pseudo=(counts.detach(),conf_sum.detach())
        self._drp_last=self.drp.diagnostics(probabilities,weights,effective,self.p_cutoff)
        class_h=head_weight.new_zeros(self.num_classes).scatter_add_(0,pseudo,head_weight)
        class_t=tail_weight.new_zeros(self.num_classes).scatter_add_(0,pseudo,tail_weight)
        total_training_weight=class_h.sum()+class_t.sum()
        self._drp_last.update(class_H_weight_sums=class_h.cpu().tolist(),class_T_weight_sums=class_t.cpu().tolist(),
            class_training_weight_sums=(class_h+class_t).cpu().tolist(),
            class_training_weight_ratio=((class_h+class_t)/total_training_weight).cpu().tolist() if total_training_weight>0 else [0.]*self.num_classes,
            hard_mask_enabled=self.args.drp_with_hard_mask,
            H_weight_sum=float(head_weight.sum()),T_weight_sum=float(tail_weight.sum()),
            weighted_unsupervised_loss=float(unsupervised.detach()))
        return supervised,unsupervised,accepted

    def train_step(self,x_lb,y_lb,y_lb_noised,x_ulb_w,x_ulb_s,y_ulb):
        # A rank-local numerical skip must not advance reliability statistics.
        self._drp_before_step=copy.deepcopy(self.drp.state_dict()) if self.drp is not None else None
        n,u=len(x_lb),len(x_ulb_w)
        def capture_weights(module,inputs,outputs):
            self._dea_last=dict(w_H_mean=float(outputs['w1'].detach().mean()),w_T_mean=float(outputs['w3'].detach().mean()),
                dynamic_w_H_mean=float(outputs['dynamic_w1'].detach().mean()),
                dynamic_w_T_mean=float(outputs['dynamic_w3'].detach().mean()),
                unlabeled_w_H_mean=float(outputs['w1'][n:n+u].detach().mean()) if u else None,
                unlabeled_w_T_mean=float(outputs['w3'][n:n+u].detach().mean()) if u else None)
        handle=self.model.register_forward_hook(capture_weights)
        try:return super().train_step(x_lb,y_lb,y_lb_noised,x_ulb_w,x_ulb_s,y_ulb)
        finally:handle.remove()

    def record_trajectory(self):
        skipped=self.log_dict.get('train/optimizer_step_skipped',0)
        if skipped and self.drp is not None:
            self.drp.load_state_dict(self._drp_before_step)
        norms={}
        for group,values in self._fusion_gradient_squares.items():
            value=torch.stack(values).sum().sqrt()/self._gradient_scale_used if values else None
            norms[group]=float(value) if value is not None and torch.isfinite(value) else 0.
        for key,value in norms.items():self.log_dict['train/'+key+'_gradient_norm']=value
        self.log_dict['train/fixed_dea_weights']=float(self.fixed_dea_weights)
        if self.epoch>self.est_epoch+1 and not skipped:
            self._mff_active_steps+=1
            self._mff_nonzero_steps+=int(norms['mff']>0)
            if (self.fixed_dea_weights and self._mff_active_steps==self.args.mff_gradient_audit_window
                    and self._mff_nonzero_steps==0):
                raise RuntimeError('M2 MFF has no effective gradients; stop formal ablation')
        super().record_trajectory()
        if self.distributed and self.rank!=0:return
        row=dict(epoch=self.epoch,iteration=self.it,drp_enabled=self.drp_enabled,
            fixed_dea_weights=self.fixed_dea_weights,drp=self._drp_last,dea_weights=self._dea_last,
            mff_gradient_norm=norms['mff'],predict_gradient_norm=norms['predict'],
            mff_active_steps=self._mff_active_steps,mff_nonzero_steps=self._mff_nonzero_steps,
            optimizer_step_skipped=bool(skipped))
        folder=Path(self.save_dir)/self.save_name;folder.mkdir(parents=True,exist_ok=True)
        with (folder/'drp_history.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')

    def protocol_settings(self):
        settings=super().protocol_settings()
        settings['drp_study']=dict(enabled=self.drp_enabled,momentum=self.args.drp_momentum,
            n_sigma=self.args.drp_n_sigma,per_class=self.args.drp_per_class,epsilon=self.args.drp_epsilon,
            hard_mask=self.args.drp_with_hard_mask,loss_epsilon=self.args.drp_loss_epsilon,
            fixed_dea_weights=self.fixed_dea_weights,mff='original_lin1_lin8',
            fusion_auxiliary_losses='original_enabled_with_original_warmup',
            mff_gradient_audit_window=self.args.mff_gradient_audit_window)
        return settings

    def get_save_dict(self):
        saved=super().get_save_dict()
        saved['drp_state']=copy.deepcopy(self.drp.state_dict()) if self.drp is not None else None
        saved['mff_gradient_monitor']=dict(active=self._mff_active_steps,nonzero=self._mff_nonzero_steps)
        return saved

    def load_model(self,load_path):
        saved=super().load_model(load_path)
        if self.drp is not None:
            if saved.get('drp_state') is None:raise ValueError('Missing DRP reliability EMA state')
            self.drp.load_state_dict(saved['drp_state'])
        monitor=saved['mff_gradient_monitor']
        self._mff_active_steps,self._mff_nonzero_steps=monitor['active'],monitor['nonzero']
        return saved

    @staticmethod
    def get_argument():
        return LLAE3MetaExpert.get_argument()+[
            SSL_Argument('--e3_drp_enabled',str2bool,True),
            SSL_Argument('--drp_momentum',float,.999),
            SSL_Argument('--drp_n_sigma',float,2.),
            SSL_Argument('--drp_per_class',str2bool,False),
            SSL_Argument('--drp_with_hard_mask',str2bool,False),
            SSL_Argument('--drp_epsilon',float,1e-8),
            SSL_Argument('--drp_loss_epsilon',float,1e-12),
            SSL_Argument('--ablation_fixed_dea_weights',str2bool,False),
            SSL_Argument('--ablation_no_mff',str2bool,False),
            SSL_Argument('--mff_gradient_audit_window',int,100),
        ]
