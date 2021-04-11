from util import flatten

import torch
from torch import nn
from torch import optim

class L1Dist(nn.Module):
    def forward(self, pred, target):
        return torch.abs(pred - target).sum()

class CosDist(nn.Module):
    def forward(self, x, y):
        nx, ny = nn.functional.normalize(x), nn.functional.normalize(y)
        return 1 - (nx * ny).sum()

class Objective(nn.Module):
    def __init__(self, vocab, repr_size, comp_fn, err_fn, zero_init):
        super().__init__()
        self.emb = nn.Embedding(len(vocab), repr_size)
        self.softmax = nn.Softmax(dim=1)
        if zero_init:
            self.emb.weight.data.zero_()
        self.comp = comp_fn
        self.err = err_fn

    def compose(self, e):
        if isinstance(e, tuple):
            args = (self.compose(ee) for ee in e)
            return self.comp(*args)
        #return self.softmax(self.emb(e))
        return self.emb(e)

    def forward(self, rep, expr):
        comp_fn = self.compose(expr)
        orig_fn = rep
        return self.err(self.compose(expr), rep), nn.functional.normalize(orig_fn).tolist(), nn.functional.normalize(comp_fn).tolist()

def evaluate(reps, exprs, comp_fn, err_fn, quiet=False, steps=400, include_pred=False, zero_init=True):
    vocab = {}
    for expr in exprs:
        toks = flatten(expr)
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    def index(e):
        if isinstance(e, tuple):
            return tuple(index(ee) for ee in e)
        return torch.LongTensor([vocab[e]])

    treps = [torch.FloatTensor([r]) for r in reps]
    texprs = [index(e) for e in exprs]
    # print(texprs[0])
    # print(treps[0])
    # print(len(treps))
    # print(len(texprs))
    #print(texprs[0].shape)
    #print(treps[0].shape)
    # print("------------DEBUG-----------")
    # print("the decoder vals are " + str(reps[0]))
    # print("the decoder vals tensors are " + str(treps[0]))
    # print("the data vals are " + str(exprs[0]))

    obj = Objective(vocab, reps[0].size, comp_fn, err_fn, zero_init)
    opt = optim.RMSprop(obj.parameters(), lr=0.00001)

    t = 0
    prev_loss = 0
    same_loss_counter = 0

    while (t < steps and same_loss_counter < 5):
        opt.zero_grad()
        errs = []
        orig_fns = []
        c_fns = []
        for r, e in zip(treps, texprs):
            err, orig_fn, c_fn = obj(r, e)
            errs.append(err)
            orig_fns.append(orig_fn)
            c_fns.append(c_fn)
        #errs = [obj(r, e) for r, e in zip(tmsgs, texprs)]
        loss = sum(errs)
        if (loss == prev_loss):
            same_loss_counter += 1
        else:
            same_loss_counter = 0
        print(loss.grad)
        loss.backward()
        print(loss.grad)
        if t == 1 or t % 100 == 0:
            print(loss.item())
        opt.step()
        t += 1
        prev_loss = loss


    # for t in range(steps):
    #     opt.zero_grad()
    #     # errs = [obj(r, e) for r, e in zip(treps, texprs)]
    #     errs = []
    #     orig_fns = []
    #     c_fns = []
    #     for r, e in zip(treps, texprs):
    #         err, orig_fn, c_fn = obj(r, e)
    #         errs.append(err)
    #         orig_fns.append(orig_fn)
    #         c_fns.append(c_fn)
    #     loss = sum(errs)
    #     # print(obj.emb.weight.grad)
    #     loss.backward()
    #     if not quiet and t % 100 == 0:
    #         print(loss.item())
    #     opt.step()

    #for r, e in zip(treps, texprs):
    #    print(r, obj.compose(e))
    #assert False
    final_errs = [err.item() for err in errs]

    emb_vals = [obj.emb(torch.LongTensor([i])) for i in range(0,len(vocab))]
    print(emb_vals)
    # print(c_fns)
    # print(orig_fns)

    return final_errs
    # print(final_errs)
    # if include_pred:
    #     lexicon = {
    #         k: obj.emb(torch.LongTensor([v])).data.cpu().numpy()
    #         for k, v in vocab.items()
    #     }
    #     composed = [obj.compose(t) for t in texprs]
    #     return final_errs, lexicon, composed
    # else:
    #     return final_errs