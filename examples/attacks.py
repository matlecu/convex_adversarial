import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from convex_adversarial import robust_loss

import numpy as np
import math, json

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def mean(l): 
    return sum(l)/len(l)


def fgs(loader, model, epsilon, verbose=False, robust=False): 
    def _fgs(model, X, y, epsilon): 
        opt = optim.Adam([X], lr=1e-3)
        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

        opt.zero_grad()
        ce.backward()
        eta = X.grad.data.sign()*epsilon
        
        X_fgs = Variable(X.data + eta)
        err_fgs = (model(X_fgs).data.max(1)[1] != y.data).float().sum()  / X.size(0)
        return err, err_fgs

    return attack(loader, model, epsilon, verbose=verbose, atk=_fgs,
                  robust=robust)


def pgd(loader, model, epsilon, niters=100, alpha=0.01, restarts=1, verbose=False,
        robust=False):
    def _pgd(model, X, y, epsilon): 
        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

        X_pgd = Variable(X.data, requires_grad=True)
        for restart in range(restarts):
            if restarts > 1:
                start = torch.empty(*X.size()).cuda()
                nn.init.uniform_(start, -1/256, +1/256)
                X_pgd = Variable(X.data + start.data, requires_grad=True)
            else:
                X_pgd = Variable(X.data, requires_grad=True)
            for i in range(niters): 
                opt = optim.Adam([X_pgd], lr=1e-3)
                opt.zero_grad()
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
                loss.backward()
                eta = alpha*X_pgd.grad.data.sign()
                X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
                
                # adjust to be within [-epsilon, epsilon]
                eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
                X_pgd = Variable(X.data + eta, requires_grad=True)
            
        #  err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum() / X.size(0)
            if restart == 0:
                err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float()
            else:
                err_pgd = err_pgd + (model(X_pgd).data.max(1)[1] != y.data).float()

        err_pgd = (err_pgd >= 1).float().sum() / X.size(0)
        return err, err_pgd

    return attack(loader, model, epsilon, verbose=verbose, atk=_pgd,
                  robust=robust)

def pgd_l2(loader, model, epsilon, niters=100, alpha=0.001, restarts=1,
        verbose=False, robust=False):
    def _pgd_l2(model, X, y, epsilon): 
        out = model(X)
        ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y.data).float().sum()  / X.size(0)

        for restart in range(restarts):
            if restarts > 1:
                start = torch.empty(*X.size()).cuda()
                nn.init.uniform_(start, -1/256, +1/256)
                X_pgd = Variable(X.data + start.data, requires_grad=True)
            else:
                X_pgd = Variable(X.data, requires_grad=True)
            for i in range(niters):
                opt = optim.Adam([X_pgd], lr=1e-3)
                opt.zero_grad()
                loss = nn.CrossEntropyLoss()(model(X_pgd), y)
                loss.backward()

                eta = alpha * nn.functional.normalize(X_pgd.grad.data)
                X_pgd = Variable(X_pgd.data + eta, requires_grad=True)

                dx_orig = X_pgd.data - X.data
                dx = dx_orig.view(dx_orig.size(0), -1)
                dx_norm = torch.norm(dx, p=2, dim=1, keepdim=True)
                dx_clipped_norm = torch.clamp(dx_norm, 0, epsilon)
                dx = (dx_clipped_norm / dx_norm) * dx
                dx_orig = dx.view(dx_orig.size())

                X_pgd = Variable(X.data + dx_orig, requires_grad=True)

            #  dx = X_pgd.data - X.data
            #  print(dx)
            #  dx = dx.view(dx.size(0), -1)
            #  dx_norm = torch.norm(dx, p=2, dim=1, keepdim=True)
            #  print(dx_norm)
            if restart == 0:
                err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float()
            else:
                err_pgd = err_pgd + (model(X_pgd).data.max(1)[1] != y.data).float()


        #  err_pgd = (model(X_pgd).data.max(1)[1] != y.data).float().sum() / X.size(0)
        err_pgd = (err_pgd >= 1).float().sum() / X.size(0)
        return err, err_pgd

    return attack(loader, model, epsilon, verbose=verbose, atk=_pgd_l2,
                  robust=robust)

def carlini_l2(loader, model, epsilon, niters=100, alpha=1e-2, restarts=1,
        verbose=False, robust=False):
    def _carlini_l2(model, X, y, epsilon): 
        #  boxmin, boxmax = -.5, .5
        boxmin, boxmax = 0., 1.

        boxmul  = (boxmax - boxmin) / 2.
        boxplus = (boxmin + boxmax) / 2.

        _X = X
        #  X = np.arctanh((_X.detach().cpu() - boxplus) / boxmul * 0.999999).cuda()
        X = np.arctanh((_X.data - boxplus) / boxmul * 0.999999).cuda()

        tlab = torch.cuda.FloatTensor(y.size(0), 10).zero_()
        tlab.scatter_(1, y.data.unsqueeze(1), 1)

        out = model(_X)
        #  ce = nn.CrossEntropyLoss()(out, y)
        err = (out.data.max(1)[1] != y.data).float().sum() / X.size(0)

        binary_search_loops = 9
        start_const = 1e-2
        const_mult  = 10

        batch_size = y.size(0)
        o_bestl2 = [1e10]*batch_size
        o_bestscore = [-1]*batch_size
        for restart in range(restarts):
            # set the lower and upper bounds accordingly
            lower_bound = np.zeros(batch_size)
            CONST = np.ones(batch_size)*start_const
            upper_bound = np.ones(batch_size)*1e10
            for _ in range(binary_search_loops):
                const  = torch.tensor(CONST).cuda().float()
                #  print("Restart {}".format(restart))
                #  print(const.data)
                bestl2 = [1e10]*batch_size
                bestscore = [-1]*batch_size
                if restart >= 1:
                    #  modifier = torch.empty(*X.size(), requires_grad=True, device="cuda")
                    #  nn.init.normal_(modifier, std=restart * 1.0)
                    _modifier = 0.05 * np.random.normal(np.zeros(X.size()), np.ones(X.size()))
                    _modifier = np.clip(_X.data.cpu().numpy() + _modifier, boxmin, boxmax)
                    _modifier = np.arctanh((_modifier - boxplus) / boxmul * 0.999999) - X.data.cpu().numpy()
                    #  modifier = Variable(modifier, requires_grad=True).cuda()
                    modifier = torch.tensor(_modifier, requires_grad=True, dtype=torch.float32, device="cuda")
                else:
                    modifier = torch.zeros(*X.size(), requires_grad=True, device="cuda")

                opt = optim.Adam([modifier], lr=alpha)
                for i in range(niters):
                    opt.zero_grad()

                    X_atk  = torch.tanh(modifier + X) * boxmul + boxplus
                    l2dist = torch.sum(((X_atk-(torch.tanh(X) * boxmul + boxplus)).pow(2)).view((X_atk.size(0), -1)), 1)
                    #  l2dist = torch.sum(((X_atk-_X).pow(2)).view([y.size(0), -1]), 0)
                    output = model(X_atk)

                    real = torch.sum(tlab.data * output, 1)
                    other = torch.max((1-tlab.data) * output - tlab.data * 10000, dim=1)[0]
                    loss1 = torch.max(torch.zeros_like(real), real - other)
                    #  exit()
                    # sum up the losses
                    loss2 = torch.sum(l2dist)
                    loss1 = torch.sum(const * loss1)
                    loss  = loss1 + loss2
   
                    loss.backward()
                    opt.step()

                    err_atk = (output.data.max(1)[1] != y.data)
                    for j, _err in enumerate(err_atk.data):
                        l2 = math.sqrt(l2dist[j].item())
                        _err = _err.item()
                        #  print(_err)
                        #  print(l2)
                        if l2 < bestl2[j] and _err > 0:
                            # Success for thie round
                            bestl2[j] = l2
                            bestscore[j] = output.data.max(1)[0][j].item()
                        if l2 < o_bestl2[j] and _err > 0:
                            #  print("Smaller attack found!")
                            # Best found so far
                            o_bestl2[j] = l2

                #  print()
                #  print(o_bestl2)
                for e in range(batch_size):
                    if bestscore[e] != -1:
                        # success, divide const by two
                        upper_bound[e] = min(upper_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        # failure, either multiply by 10 if no solution found yet
                        #          or do binary search with the known upper bound
                        lower_bound[e] = max(lower_bound[e], CONST[e])
                        if upper_bound[e] < 1e9:
                            CONST[e] = (lower_bound[e] + upper_bound[e])/2
                        else:
                            CONST[e] *= 10


        err_pgd = o_bestl2
        #  print(err_pgd)
        return [err.item()], err_pgd

    return attack(loader, model, epsilon, verbose=verbose, atk=_carlini_l2,
                  robust=robust)

def attack(loader, model, epsilon, verbose=False, atk=None,
           robust=False):
    
    total_err, total_fgs, total_robust = [],[],[]
    if verbose: 
        print("Requiring no gradients for parameters.")
    for p in model.parameters(): 
        p.requires_grad = False
    
    tot_images = 0
    for i, (X,y) in enumerate(loader):
        print("{} images processed".format(tot_images))
        if tot_images >= 2000:
            break
        tot_images += len(X)
        X,y = Variable(X.cuda(), requires_grad=True), Variable(y.cuda().long())

        if y.dim() == 2: 
            y = y.squeeze(1)

        if robust: 
            robust_ce, robust_err = robust_loss_batch(model, epsilon, X, y, False, False)

        err, err_fgs = atk(model, X, y, epsilon)

        total_err.append(err)
        total_fgs.append(err_fgs)
        if robust: 
            total_robust.append(robust_err)
        if verbose: 
            if robust: 
                print('err: {} | attack: {} | robust: {}'.format(err, err_fgs, robust_err))
            else:
                print('err: {} | attack: {}'.format(err, err_fgs))

    with open("carlini_attack_svhn.json", "w") as f:
        f.write(json.dumps({'err': total_err, 'attack_err': total_fgs}))

    if robust:
        print('[TOTAL] err: {} | attack: {} | robust: {}'.format(mean(total_err), mean(total_fgs), mean(total_robust)))
    else:
        print('[TOTAL] err: {} | attack: {}'.format(mean(total_err), mean(total_fgs)))
    return mean(total_err), mean(total_fgs), total_robust

