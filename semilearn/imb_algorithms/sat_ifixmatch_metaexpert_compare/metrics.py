from __future__ import annotations
import numpy as np

def _to_ood(v,k): return k if int(v)<0 or int(v)>=k else int(v)

def confusion_matrix_with_ood(y_true,y_pred,num_classes):
    k=num_classes; cm=np.zeros((k+1,k+1),dtype=np.int64)
    for t,p in zip(y_true,y_pred): cm[_to_ood(t,k),_to_ood(p,k)]+=1
    return cm

def accuracy_known(y_true,y_pred,num_classes):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); m=(y_true>=0)&(y_true<num_classes)
    return float((y_true[m]==y_pred[m]).mean()) if m.sum() else 0.0

def overall_accuracy_with_ood(y_true,y_pred,num_classes):
    y_true=np.asarray(y_true).copy(); y_pred=np.asarray(y_pred).copy()
    y_true[(y_true<0)|(y_true>=num_classes)]=-1; y_pred[(y_pred<0)|(y_pred>=num_classes)]=-1
    return float((y_true==y_pred).mean())

def macro_f1_known(y_true,y_pred,num_classes,eps=1e-12):
    cm=confusion_matrix_with_ood(y_true,y_pred,num_classes)[:num_classes,:num_classes]
    vals=[]
    for c in range(num_classes):
        tp=cm[c,c]; fp=cm[:,c].sum()-tp; fn=cm[c,:].sum()-tp
        p=tp/(tp+fp+eps); r=tp/(tp+fn+eps); vals.append(2*p*r/(p+r+eps))
    return float(np.mean(vals))

def per_class_recall_known(y_true,y_pred,num_classes,eps=1e-12):
    cm=confusion_matrix_with_ood(y_true,y_pred,num_classes)[:num_classes,:num_classes]
    return [float(cm[c,c]/(cm[c,:].sum()+eps)) for c in range(num_classes)]

def gmean_known(y_true,y_pred,num_classes,eps=1e-12):
    rec=np.asarray(per_class_recall_known(y_true,y_pred,num_classes,eps))
    return float(np.exp(np.mean(np.log(rec+eps))))

def odr(y_true,y_pred,num_classes,eps=1e-12):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); m=(y_true<0)|(y_true>=num_classes)
    if m.sum()==0: return None
    det=(y_pred<0)|(y_pred>=num_classes)
    return float((det&m).sum()/(m.sum()+eps))

def fdr(y_true,y_pred,num_classes,eps=1e-12):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); m=(y_true>=0)&(y_true<num_classes)
    if m.sum()==0: return None
    det=(y_pred<0)|(y_pred>=num_classes)
    return float((det&m).sum()/(m.sum()+eps))

def summarize(y_true,y_pred,num_classes):
    return {'known_accuracy':accuracy_known(y_true,y_pred,num_classes),'overall_accuracy_with_ood':overall_accuracy_with_ood(y_true,y_pred,num_classes),'macro_f1_known':macro_f1_known(y_true,y_pred,num_classes),'gmean_known':gmean_known(y_true,y_pred,num_classes),'per_class_recall_known':per_class_recall_known(y_true,y_pred,num_classes),'odr':odr(y_true,y_pred,num_classes),'fdr':fdr(y_true,y_pred,num_classes),'confusion_matrix_with_ood':confusion_matrix_with_ood(y_true,y_pred,num_classes).tolist()}
