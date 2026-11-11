with open("eval_confusion_matrix.py", "r", encoding="utf-8") as f:
    text = f.read()

injection = """    net = algorithm.ema_model.to(device)
    
    # Check if MetaExpert attributes need to be moved to device
    if hasattr(net, 'model'):
        m = net.model
    else:
        m = net
    for attr in ['hat', 'tau1', 'tau2', 'tau3']:
        if hasattr(m, attr):
            t = getattr(m, attr)
            if isinstance(t, torch.Tensor):
                setattr(m, attr, t.to(device))
"""
text = text.replace("    net = algorithm.ema_model.to(device)", injection)

with open("eval_confusion_matrix.py", "w", encoding="utf-8") as f:
    f.write(text)
