import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from convex_adversarial import robust_loss

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

def attack(loader, model, epsilon, verbose=False, atk=None,
           robust=False):
    
    total_err, total_fgs, total_robust = [],[],[]
    if verbose: 
        print("Requiring no gradients for parameters.")
    for p in model.parameters(): 
        p.requires_grad = False
    
    for i, (X,y) in enumerate(loader):
        if i >= 1000:
            break
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
    
    if robust:         
        print('[TOTAL] err: {} | attack: {} | robust: {}'.format(mean(total_err), mean(total_fgs), mean(total_robust)))
    else:
        print('[TOTAL] err: {} | attack: {}'.format(mean(total_err), mean(total_fgs)))
    return mean(total_err), mean(total_fgs), total_robust

