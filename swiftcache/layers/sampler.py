import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, top_ps=None):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        if bool(torch.all(temperatures == 0)):
            return greedy_tokens
        logits.div_(temperatures.clamp_min(1e-6).unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        if top_ps is not None and bool(torch.any(top_ps < 1)):
            ordered,indices = probs.sort(dim=-1,descending=True)
            ordered.masked_fill_(ordered.cumsum(-1)-ordered >= top_ps.unsqueeze(1),0)
            selected=ordered.div(torch.empty_like(ordered).exponential_(1).clamp_min(1e-10)).argmax(-1)
            sampled=indices.gather(1,selected.unsqueeze(1)).squeeze(1)
            return torch.where(temperatures==0,greedy_tokens,sampled)
        # logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)
        epsilon = 1e-10  
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(dim=-1) 
        
        res = torch.where(temperatures == 0, greedy_tokens, sample_tokens) 
        return res
