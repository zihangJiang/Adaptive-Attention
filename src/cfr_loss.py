# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
default_device = 'cuda:0'
variable_cache = {}
def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encore with respect to the last axis
    one_hot = torch.FloatTensor(tensor.size() + (n,)).zero_()
    if tensor.is_cuda : one_hot = one_hot.cuda()
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot

class SoftCrossEntropyLoss(Module):
    def __init__(self, size_average=False):
        super(SoftCrossEntropyLoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target):
        pred = F.log_softmax(input, dim=1)
        tgt_label = F.softmax(target, dim=1)
        pre_ent = - tgt_label * pred
        ent = pre_ent.sum(1)
        if self.size_average:
            return ent.mean()
        else:
            return ent.sum()
class EntropyLoss(Module):
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0*b.sum()#(dim = 1)
        return torch.mean(b)

class CFRLoss(Module):
    '''
    Loss class deriving from Module for the cfr loss function defined below
    '''
    def __init__(self, class_per_it):
        super(CFRLoss, self).__init__()
        self.class_per_it = class_per_it

    def forward(self, input, target, class_per_it,num_support):
        return cfr_loss(input, target, class_per_it,num_support)
    
def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def cfr_loss(input, target, class_per_it, num_support, device = default_device):
    '''
    
    Compute the cross entropy of result input and target, 
    loss and accuracy are then computed and returned
    Args:
    - input: the model output for a batch of samples
    - target: ground truth for the above batch of samples
    '''
    classes = target[:class_per_it*num_support].view(class_per_it, num_support)[:,0]


    if not input.size(0)==target.size(0):
        target_idx = target[class_per_it*num_support:]
    else:
        target_idx = target
    target_idx = label_to_index(classes, target_idx)
    loss_val = CrossEntropyLoss()(input, target_idx)
    y_hat = torch.argmax(input, dim = 1)

    acc_val = y_hat.eq(target_idx).float().mean()

    return loss_val,  acc_val

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits
def label_to_index(label, seq):
    n_classes = len(label)
    num_samples = len(seq)
    if (n_classes, num_samples) not in variable_cache:
        idxs=torch.arange(n_classes).repeat(num_samples, 1).transpose(0, 1).to(default_device)
        variable_cache[(n_classes, num_samples)] = idxs
    else:
        idxs = variable_cache[(n_classes, num_samples)]
    labels = label.repeat(num_samples, 1).transpose(0, 1)
    seqs = seq.repeat(n_classes, 1)
    idx = torch.where(seqs==labels, idxs, torch.zeros_like(idxs))
    return idx.sum(dim=0)
    
def ensembled_loss(input, perm_input, input_weight, reference_weight, target, class_per_it, num_support, device = default_device):
    '''
    Compute the cross entropy of result (input, perm_input) with target, 
    combined with the inter class loss of the reweighting coeff
    loss and accuracy are then computed and returned
    Args:
    - input: the model output for a batch of samples
    - perm_input: the model perm output for a batch of samples
    - target: ground truth for the above batch of samples
    '''

    classes = target[:class_per_it*num_support].view(class_per_it, num_support)[:,0]
    if not input.size(0)==target.size(0):
        target_idx = target[class_per_it*num_support:]
    else:
        target_idx = target

    target_idx = label_to_index(classes, target_idx)

    loss_val = CrossEntropyLoss()(input, target_idx)

    perm_loss_val = CrossEntropyLoss()(perm_input, target_idx)

    ensemble_loss_val = CrossEntropyLoss()(input + perm_input, target_idx)

    entropy_loss = EntropyLoss()(input) + EntropyLoss()(perm_input)

    ensemble_loss_val += 0.01*entropy_loss

    y_hat = torch.argmax(input, dim = 1)
    perm_y_hat = torch.argmax(perm_input, dim = 1)

    acc_val = y_hat.eq(target_idx).float().mean()
    perm_acc_val = perm_y_hat.eq(target_idx).float().mean()
    
    logits = euclidean_metric(input_weight, reference_weight)
    inter_class_loss = F.cross_entropy(logits, target_idx)

    return loss_val,  acc_val ,perm_loss_val, perm_acc_val, inter_class_loss, ensemble_loss_val

def test_loss(input, perm_input, input_weight, reference_weight, target, class_per_it, num_support, device = default_device):
    '''
    Compute the cross entropy of result (input, perm_input) with target, 
    combined with the inter class loss of the reweighting coeff
    loss and accuracy are then computed and returned
    Args:
    - input: the model output for a batch of samples
    - perm_input: the model perm output for a batch of samples
    - target: ground truth for the above batch of samples
    '''
    classes = target[:class_per_it*num_support].view(class_per_it, num_support)[:,0]
    if not input.size(0)==target.size(0):
        target_idx = target[class_per_it*num_support:]
    else:
        target_idx = target

    target_idx = label_to_index(classes, target_idx)

    loss_val = CrossEntropyLoss()(input, target_idx)

    perm_loss_val = CrossEntropyLoss()(perm_input, target_idx)

    ensemble_loss_val = CrossEntropyLoss()(input + perm_input, target_idx)
    


    y_hat = torch.argmax(input, dim = 1)
    perm_y_hat = torch.argmax(perm_input, dim = 1)
    ensemble_y_hat = torch.argmax(perm_input+input, dim = 1)

    acc_val = y_hat.eq(target_idx).float().mean()
    perm_acc_val = perm_y_hat.eq(target_idx).float().mean()
    ensemble_acc_val = ensemble_y_hat.eq(target_idx).float().mean()


    return loss_val,  acc_val ,perm_loss_val, perm_acc_val, ensemble_loss_val, ensemble_acc_val

def proto_loss(input_weight, reference_weight, target, class_per_it, num_support, device = default_device):
    '''
    Compute the cross entropy of result distance:
    euclidean_metric(input_weight, reference_weight) with target, 
    loss and accuracy are then computed and returned
    Args:
    - input_weight: the feature for a batch of query samples
    - reference_weight: the feature for a batch of support set
    - target: ground truth for the above batch of query samples
    '''
    classes = target[:class_per_it*num_support].view(class_per_it, num_support)[:,0]
    if not input_weight.size(0)==target.size(0):
        target_idx = target[class_per_it*num_support:]
    else:
        target_idx = target

    target_idx = label_to_index(classes, target_idx)
    logits = euclidean_metric(input_weight, reference_weight)
    y_hat = torch.argmax(logits, dim = 1)
    acc_val = y_hat.eq(target_idx).float().mean()
    loss_val = F.cross_entropy(logits, target_idx)

    return loss_val,  acc_val
def relation_loss(input, target, class_per_it, num_support, device = default_device):
    '''
    Compute the cross entropy of result distance:
    euclidean_metric(input_weight, reference_weight) with target, 
    loss and accuracy are then computed and returned
    Args:
    - input_weight: the feature for a batch of query samples
    - reference_weight: the feature for a batch of support set
    - target: ground truth for the above batch of query samples
    '''
    classes = target[:class_per_it*num_support].view(class_per_it, num_support)[:,0]
    if not input.size(0)==target.size(0):
        target_idx = target[class_per_it*num_support:]
    else:
        target_idx = target

    target_idx = label_to_index(classes, target_idx)
    y_hat = torch.argmax(input, dim = 1)
    acc_val = y_hat.eq(target_idx).float().mean()
    loss_val = F.cross_entropy(input, target_idx)

    return loss_val,  acc_val