"""Optional PyTorch head; no model download or modification of Falcon.
Inputs: h512 + pointgoal2 + remaining_fraction1 + previous_action5 + candidate4.
Previous action has a separate reset sentinel. All normalization is fit-only.
"""
import torch
from torch import nn

class TemporalHead(nn.Module):
    def __init__(self, horizon=32, width=128):
        super().__init__()
        if horizon<1 or width<1:raise ValueError('Positive dimensions required')
        self.horizon=horizon
        self.net=nn.Sequential(nn.Linear(524,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,horizon*5))
    def forward(self, x):
        if x.shape[-1]!=524:raise ValueError('Expected 524 legal input features')
        return self.net(x).reshape(*x.shape[:-1],self.horizon,5)

def censored_loss(logits, steps, events):
    """events=0 means censored, 1..4 are mutually exclusive first events.
    steps in [0,H]; terminal steps must be positive. Batch mean includes zero exposures.
    """
    if logits.ndim!=3 or logits.shape[-1]!=5:raise ValueError('Expected B,H,5')
    b,h,_=logits.shape
    if steps.shape!=(b,) or events.shape!=(b,) or steps.dtype!=torch.long or events.dtype!=torch.long:
        raise ValueError('Expected int64 B labels')
    if not torch.isfinite(logits).all() or ((steps<0)|(steps>h)|(events<0)|(events>4)|((events>0)&(steps==0))).any():raise ValueError('Invalid labels/logits')
    t=torch.arange(h,device=logits.device)[None,:]
    target=torch.where(t==steps[:,None]-1,events[:,None],0)
    loss=-logits.log_softmax(-1).gather(-1,target.unsqueeze(-1)).squeeze(-1)
    return (loss*(t<steps[:,None])).sum(-1).mean()
