import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from copy import deepcopy
def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    source: https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer
    
    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''    
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = int((h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2)
        w_pad = int((w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if(i == 0):
            spp = x.view(num_sample,-1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp,x.view(num_sample,-1)), 1)
    return spp
def base_block(in_channels, out_channels, kernel_size = 3, padding = 1):
    '''
    returns a block conv-bn
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels)
    )
class ResBlock(nn.Module):
    '''
    returns a 3 conv res block with max pooling 
    '''
    def __init__(self, in_channels, out_channels, stride = 2, padding = 0, pooling = 2):
        super(ResBlock, self).__init__()

        self.conv1 = base_block(in_channels, out_channels)
        self.conv2 = base_block(out_channels, out_channels)
        self.conv3 = base_block(out_channels, out_channels)
        self.short_conv = base_block(in_channels, out_channels,1,0)
        self.sig = nn.Sigmoid()
        self.max_pool = nn.MaxPool2d(pooling, stride = stride, padding = padding)

    def forward(self, x):
        out = self.conv1(x)
        out = self.sig(out)*out
        out = self.conv2(out)
        out = self.sig(out)*out
        out = self.conv3(out)
        out += self.short_conv(x)
        out = self.sig(out)*out
        out = self.max_pool(out)
        return out
def conv_block(in_channels, out_channels, stride = 2, padding = 0, pooling = 2):
    '''
    returns a block conv-bn-relu-pool
    '''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),#, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(pooling, stride = stride, padding = padding)
    )
def linear_block(in_channels, out_channels):
    '''
    returns a block linear-bn-relu
    '''
    return nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.LeakyReLU(0.1)
    )
def channel_wise_mul_per_batch(inputs, weight):
    '''
    inputs: (N, C, H, W)
    weight: (n_cls, C)
    outputs: (N*n_cls, C, H, W)
    multiply weight and feature maps channel wisely and repeatedly.
    i.e. each weight of one class will multiply each input once.
    [f1, f2, f3, ..., fn]*[w1, w2, w3, ..., wn] = [f1*w1, f2*w2, ..., fn*wn]
    '''
    n_channel = inputs.size(1)
    n_cls = weight.size(0)
    assert weight.size(1) == n_channel
    
    # Get group size
    group_size = 1
    
    groups = n_cls* n_channel // group_size
    # Reshape input tensor (N, C, H, W) to (N, n_cls*C, H, W)
    inputs = inputs.repeat(1, n_cls, 1, 1)
    # Reshape weight tensor from size (n_cls, C) to (n_cls*C, 1, 1, 1)
    weight = weight.view(-1, group_size , 1, 1)
    # get result of (N, n_cls*C, H, W)
    conv_result = F.conv2d(inputs, weight, groups = groups)
    return conv_result.view(-1, n_channel, conv_result.size(2), conv_result.size(3))
def channel_wise_bmm(inputs, weight):
    '''
    inputs: (N, C, H, W)
    weight: (N, C)
    outputs: (N, C, H, W)
    multiply weight and feature maps channel wisely and batch wisely.
    [f1, f2, f3, ..., fn]*[w1, w2, w3, ..., wn] = [f1*w1, f2*w2, ..., fn*wn]
    '''
    n_channel = inputs.size(1)
    N = weight.size(0)
    assert weight.size(1) == n_channel
    group_size = 1
    groups = N* n_channel // group_size

    # Reshape input tensor (N, C, H, W) to (1, N*C, H, W)
    inputs = inputs.view(1, N*n_channel, inputs.size(2), inputs.size(3))
    # Reshape weight tensor from size (N, C) to (N*C, 1, 1, 1)
    weight = weight.view(-1, group_size , 1, 1)
    # get result of (1, N*C, H, W)
    conv_result = F.conv2d(inputs, weight, groups = groups)
    return conv_result.view(-1, n_channel, conv_result.size(2), conv_result.size(3))
class L2Norm(nn.Module):
    def __init__(self):
        super(L2Norm, self).__init__()
    def forward(self, x):
        return x/torch.sqrt(torch.sum(x*x,1).view(x.size(0),1))
class ProtoNet(nn.Module):
    '''
    Feature exractor model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    input: source image (or image belong to a certain class) (N*C*H*W)
    output: batch_size of feature maps (N*C*iH*iW)
    '''
    def __init__(self, opt, x_dim=3, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=opt.scale)
        if opt.dataset in ['MiniImagenet', 'CUB', 'StanfordCar', 'StanfordDog']:
            if opt.resnet:
                self.FeatureExtractor = nn.Sequential(
                    ResBlock(x_dim, hid_dim),
                    ResBlock(hid_dim, int(hid_dim*1.5)),
                    ResBlock(int(hid_dim*1.5), hid_dim*2),
                    ResBlock(hid_dim*2, z_dim, 1, 1, 3)
                    )
            else:
                self.FeatureExtractor = nn.Sequential(
                    conv_block(x_dim, hid_dim),
                    conv_block(hid_dim, hid_dim),
                    conv_block(hid_dim, hid_dim),
                    conv_block(hid_dim, z_dim),
                )# output will be (btz,64,5,5) for miniimagenet

        elif opt.dataset == 'Omniglot':
            x_dim = 1
            self.FeatureExtractor = nn.Sequential(
                conv_block(x_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, z_dim),
            )# output will be (btz,64,1,1) for omniglot
    def forward(self, x):
        x = self.FeatureExtractor(x)
        x = x*self.scale_cls
        return x.view(x.size(0),-1)


class ANet(nn.Module):
    '''
    attention module
    input: batch_size*n_classes of new feature maps(NC*iH*iW)
    output: score \in [0,1] for a certain class corresponding to the weight (N*Ncls)
    '''
    def __init__(self, opt, x_dim=64, hid_dim=64, z_dim=1):
        super(ANet, self).__init__()
        self.scale_atten = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=opt.scale)
        if opt.simplest:
            self.Attention = nn.Sequential(
                nn.Conv2d(x_dim, z_dim, 1),
                nn.ReLU()
                )
        else:
            self.Attention = nn.Sequential(
                nn.Conv2d(x_dim, hid_dim, 3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(hid_dim, z_dim, 3, padding=1),
                nn.ReLU()
                )
    def forward(self, x):
        x = self.Attention(x)
        x = self.scale_atten*torch.mean(torch.mean(x,dim=2),dim=2).view(-1)
        return x
    
class RNet(nn.Module):
    '''
    Reweighting net
    input: batch_size of feature maps (N*C*iH*iW)
    output: weight for feature reweighting(N*kW)
    '''
    def __init__(self, opt, x_dim=64, hid_dim=128, z_dim=64, output_num = [2,1]):
        super(RNet, self).__init__()
        self.output_num = output_num
        self.scale_weight = 1.0#nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)
        if opt.dataset in ['MiniImagenet', 'CUB', 'StanfordCar', 'StanfordDog']:
            self.Reweighter = nn.Sequential(
                linear_block(x_dim, hid_dim),
                linear_block(hid_dim, hid_dim),
                nn.Linear(hid_dim, z_dim)
            )
        else:
            self.Reweighter = nn.Sequential(
                linear_block(x_dim, hid_dim),
                linear_block(hid_dim, hid_dim),
                nn.Linear(hid_dim, z_dim),
            )
    def forward(self, x):
        x = spatial_pyramid_pool(x,x.size(0),[x.size(2),x.size(3)],self.output_num)
        x = self.Reweighter(x)
        x = x/torch.sqrt(torch.sum(x*x,1).view(x.size(0),1))
        x = self.scale_weight*x
        return x


class FNet(nn.Module):
    '''
    Feature exractor model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    input: source image (or image belong to a certain class) (N*C*H*W)
    output: batch_size of feature maps (N*C*iH*iW)
    '''
    def __init__(self, opt, x_dim=1, hid_dim=64, z_dim=64):
        super(FNet, self).__init__()
        if opt.dataset in ['MiniImagenet', 'CUB', 'StanfordCar', 'StanfordDog']:
            if opt.resnet:
                self.FeatureExtractor = nn.Sequential(
                    ResBlock(x_dim, hid_dim),
                    ResBlock(hid_dim, int(hid_dim*1.5)),
                    ResBlock(int(hid_dim*1.5), hid_dim*2),
                    ResBlock(hid_dim*2, z_dim, 1, 1, 3)
                    )
            else:
                self.FeatureExtractor = nn.Sequential(
                    conv_block(x_dim, hid_dim),
                    conv_block(hid_dim, hid_dim),
                    conv_block(hid_dim, hid_dim),
                    conv_block(hid_dim, z_dim, 1, 1, 3),
                )# output will be (btz,64,10,10) for miniimagenet

        elif opt.dataset == 'Omniglot':
            self.FeatureExtractor = nn.Sequential(
                conv_block(x_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, hid_dim, 1, 1, 3),
                conv_block(hid_dim, z_dim, 1, 1, 3),
            )# output will be (btz,64,6,6) for omniglot
    def forward(self, x):
        x = self.FeatureExtractor(x)
        return x



class CANet(nn.Module):
    '''
    Classifier with attention
    input: batch_size*n_classes of new feature maps(NC*iH*iW), batch_size of attention maps of (1*mH*mW)
    output: score \in [0,1] for a certain class corresponding to the weight (N*Ncls)
    '''
    def __init__(self, opt, x_dim=64, hid_dim=128, z_dim=1, output_num = [4,2,1]):
        super(CANet, self).__init__()
        self.output_num = output_num
        # TODO
        self.thresh = nn.Threshold(0, 0)
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=opt.scale)
        if opt.dataset == 'MiniImagenet':
            self.Classifier = nn.Sequential(
                linear_block(x_dim, hid_dim),
                linear_block(hid_dim, hid_dim),
                nn.Linear(hid_dim, z_dim),
            )
        elif opt.dataset in ['StanfordDog', 'StanfordCar', 'CUB']:
            self.Classifier = nn.Sequential(
                linear_block(x_dim, hid_dim),
                linear_block(hid_dim, hid_dim),
                nn.Linear(hid_dim, z_dim),
            )
        elif opt.dataset == 'Omniglot':
            self.Classifier = nn.Sequential(
                linear_block(x_dim, hid_dim),
                # linear_block(hid_dim, hid_dim),
                nn.Linear(hid_dim, z_dim),
            )

    def forward(self, x, atten_maps):
        # comment to backprop through atten_maps
        # atten_maps = deepcopy(atten_maps.detach())
        atten_shape = atten_maps.size()

        # batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        # batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        # atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,))-batch_mins, batch_maxs - batch_mins)
        # atten_normed = atten_normed.view(atten_shape)
        atten_normed = atten_maps
        atten_threshed = self.thresh(atten_normed)
        x = x*atten_threshed
        x = spatial_pyramid_pool(x,x.size(0),[x.size(2),x.size(3)],self.output_num)
        x = x.view(x.size(0), -1)
        x = self.Classifier(x)*self.scale_cls
        return x


class RelationNet(nn.Module):
    '''
    Relation Network model as described in the reference paper,
    source: https://github.com/floodsung/LearningToCompare_FSL
    input: source image (or image belong to a certain class) (N*C*H*W)
    output: batch_size of feature maps (N*C*iH*iW)
    '''
    def __init__(self, opt, x_dim=3, hid_dim=64, z_dim=64):
        super(RelationNet, self).__init__()
        self.classes_tr = opt.classes_per_it_tr
        self.classes_val = opt.classes_per_it_val
        self.num_support_tr = opt.num_support_tr * opt.augment
        self.num_support_val = opt.num_support_val * opt.augment
        self.feature_dim = z_dim
        self.full_load = True
        if opt.dataset in ['MiniImagenet', 'CUB', 'StanfordCar', 'StanfordDog']:
            if opt.resnet:
                self.FeatureExtractor = nn.Sequential(
                    ResBlock(x_dim, hid_dim),
                    ResBlock(hid_dim, int(hid_dim*1.5)),
                    ResBlock(int(hid_dim*1.5), hid_dim*2),
                    ResBlock(hid_dim*2, z_dim, 1, 1, 3)
                    )
            else:
                self.FeatureExtractor = nn.Sequential(
                    conv_block(x_dim, hid_dim),
                    conv_block(hid_dim, hid_dim),
                    conv_block(hid_dim, hid_dim),
                    conv_block(hid_dim, z_dim, 1, 1, 3),
                )# output will be (btz,64,5,5) for miniimagenet
        elif opt.dataset == 'Omniglot':
            x_dim = 1
            self.FeatureExtractor = nn.Sequential(
                conv_block(x_dim, hid_dim),
                conv_block(hid_dim, hid_dim),
                conv_block(hid_dim, hid_dim, 1, 1, 3),
                conv_block(hid_dim, z_dim, 1, 1, 3),
            )# output will be (btz,64,1,1) for omniglot
        self.a = ANet(opt, 2*z_dim, hid_dim, 1)
        if self.full_load:
            linear_hidden_dim = 200
            output_num = [2,1]
            self.ca = CANet(opt, z_dim*sum([xx**2 for xx in output_num]), linear_hidden_dim, 1, output_num)
    def forward(self, x, s):
        nQuery = x.size(0)
        nRef = s.size(0)
        if nRef==self.classes_tr*self.num_support_tr:
            classes = self.classes_tr
            num_support = self.num_support_tr
        else:
            classes = self.classes_val
            num_support = self.num_support_val

        whole = torch.cat((x,s), dim = 0)
        feature_maps = self.FeatureExtractor(whole)
        x = feature_maps[:nQuery]
        perm_x = feature_maps[nQuery:]
        # only support one shot
        # assert num_support==1
        # inspired by https://github.com/floodsung/LearningToCompare_FSL
        perm_x = perm_x.view((classes, num_support)+perm_x.size()[1:]).mean(1)
        perm_x_ext = perm_x.unsqueeze(0).repeat(nQuery,1,1,1,1)

        x_ext = x.unsqueeze(0).repeat(classes,1,1,1,1)
        x_ext = torch.transpose(x_ext,0,1)
        relation_pairs = torch.cat((perm_x_ext,x_ext),2).view((-1,2*self.feature_dim)+x_ext.size()[3:])



        atten_result = self.a(relation_pairs)
        if self.full_load:
            perm_relation_pairs = torch.cat((x_ext, perm_x_ext),2).view((-1,2*self.feature_dim)+x_ext.size()[3:])
            atten_perm_result = self.a(perm_relation_pairs)

            atten_maps = self.a.Attention(relation_pairs)
            perm_atten_maps = self.a.Attention(perm_relation_pairs)

            perm_result = self.ca(perm_x_ext.view((-1,)+perm_x_ext.size()[2:]), perm_atten_maps)
            result = self.ca(x_ext.contiguous().view((-1,)+x_ext.size()[2:]), atten_maps)

            perm_result = perm_result.view(perm_x.size(0),-1).transpose(0,1)
            # perm_result = perm_result.view(perm_result.size(0),classes, num_support).mean(dim = 2)

            atten_perm_result = atten_perm_result.view(perm_x.size(0),-1).transpose(0,1)
            # atten_perm_result = atten_perm_result.view(atten_perm_result.size(0),classes, num_support).mean(dim = 2)

            result = result.view(x.size(0), -1)
            atten_result = atten_result.view(x.size(0), -1)
            perm_x = perm_x.view(perm_x.size(0),-1)
            # perm_x = perm_x.view(classes, num_support, perm_x.size(1)).mean(dim = 1)
            return result, perm_result, atten_result, atten_perm_result, perm_x, x.view(x.size(0),-1)
        else:
            return atten_result.view(x.size(0), -1)

    def get_cam(self, x, s):
        '''
        get class activation maps on query image
        '''
        
        nRef = s.size(0)
        nQuery = x.size(0)

        classes = self.classes_val
        num_support = self.num_support_val


        whole = torch.cat((x,s), dim = 0)
        feature_maps = self.FeatureExtractor(whole)
        x = feature_maps[:nQuery]
        perm_x = feature_maps[nQuery:]
        perm_x = perm_x.view((classes, num_support)+perm_x.size()[1:]).sum(1)
        perm_x_ext = perm_x.unsqueeze(0).repeat(nQuery,1,1,1,1)

        x_ext = x.unsqueeze(0).repeat(classes,1,1,1,1)
        x_ext = torch.transpose(x_ext,0,1)
        relation_pairs = torch.cat((perm_x_ext,x_ext),2).view((-1,2*self.feature_dim)+x_ext.size()[3:])
        cams = self.a.Attention(relation_pairs)
        
        cams_reshape = cams.view(x.size(0),-1)
        batch_maxs, _ = torch.max(cams_reshape, dim=-1, keepdim=True)
        batch_mins, _ = torch.min(cams_reshape, dim=-1, keepdim=True)
        cam_normed = torch.div(cams_reshape-batch_mins, batch_maxs - batch_mins)
        cam_normed = cam_normed.view(cams.size())
        return cam_normed
        # return cams

class CRFNet(nn.Module):
    '''
    combined model to classify via attentive feature map reweighting
    input: source image (or image belong to a certain class)
    output: batch_size of class score vector for each input image (N*Ncls)
    '''
    def __init__(self, opt, n_gpu = 1):
        super(CRFNet, self).__init__()
        if opt.dataset == 'Omniglot':
            linear_hidden_dim = 128
            hidden_dim = 64
            num_maps = 64
            image_channel = 1
            output_num = [2,1]
        elif opt.dataset == 'MiniImagenet':
            linear_hidden_dim = 200#50#
            hidden_dim = 64
            
            if opt.resnet:
                num_maps = 64*4
                output_num = [2,1]
            else:
                num_maps = 64
                output_num = [2,1]
            image_channel = 3
        elif opt.dataset in ['StanfordDog', 'StanfordCar', 'CUB']:
            linear_hidden_dim = 200
            hidden_dim = 64
            
            if opt.resnet:
                num_maps = 64*4
                output_num = [2,1]
            else:
                num_maps = 64
                output_num = [2,1]
            image_channel = 3
        self.feature_mode = opt.feature_mode
        self.n_gpu = n_gpu
        self.classes_tr = opt.classes_per_it_tr
        self.classes_val = opt.classes_per_it_val
        self.num_support_tr = opt.num_support_tr * opt.augment
        self.num_support_val = opt.num_support_val * opt.augment
        self.double = opt.double
        self.a = ANet(opt, num_maps, hidden_dim, 1)
        # weight for attention
        self.r = RNet(opt, num_maps*sum([xx**2 for xx in output_num]), linear_hidden_dim, num_maps, output_num)
        self.f = FNet(opt, image_channel, hidden_dim, num_maps)
        if self.double:
            assert not self.feature_mode=='unweighted'
            # new weight for classify
            self.rc = RNet(opt, num_maps*sum([xx**2 for xx in output_num]), linear_hidden_dim, num_maps, output_num)

        if self.feature_mode == 'weighted' or self.feature_mode == 'unweighted':
            self.ca = CANet(opt, num_maps*sum([xx**2 for xx in output_num]), linear_hidden_dim, 1, output_num)
        else:
            assert self.feature_mode=='cat'
            self.ca = CANet(opt, 2*num_maps*sum([xx**2 for xx in output_num]), linear_hidden_dim, 1, output_num)

    def forward(self, x, s):
        nQuery = x.size(0)
        nRef = s.size(0)
        whole = torch.cat((x,s), dim = 0)
        # compute feature
        feature_maps = self.f(whole)
        # calculate weight
        weight = self.r(feature_maps)

        x = feature_maps[:nQuery]
        perm_x = feature_maps[nQuery:]
        
        perm_w = weight[:nQuery]
        w = weight[nQuery:]

        if nRef==self.classes_tr*self.num_support_tr:
            classes = self.classes_tr
            num_support = self.num_support_tr
        else:
            classes = self.classes_val
            num_support = self.num_support_val
        w = w.view(classes, num_support, w.size(1)).mean(dim = 1)
        new_perm_feature_maps = channel_wise_mul_per_batch(perm_x, perm_w)
        # nQuery * nRef
        nPermMaps = new_perm_feature_maps.size(0)
        
        new_feature_maps = channel_wise_mul_per_batch(x,w)
        maps = torch.cat((new_perm_feature_maps, new_feature_maps), dim = 0)
        atten_maps = self.a.Attention(maps)
        atten_results = self.a.scale_atten*torch.mean(torch.mean(atten_maps,dim=2),dim=2).view(-1)
        if self.double:
            weight = self.rc(feature_maps)
            perm_w = weight[:nQuery]
            w = weight[nQuery:]
            w = w.view(classes, num_support, w.size(1)).mean(dim = 1)
            new_perm_feature_maps = channel_wise_mul_per_batch(perm_x, perm_w)
            new_feature_maps = channel_wise_mul_per_batch(x,w)

            maps = torch.cat((new_perm_feature_maps, new_feature_maps), dim = 0)


        if not self.feature_mode == 'weighted':
            unweighted_maps = torch.cat((
                perm_x.repeat(1,nQuery,1,1).view((-1,)+perm_x.size()[1:]),
                x.repeat(1,classes,1,1).view((-1,)+x.size()[1:])), dim = 0)
            if self.feature_mode == 'unweighted':
                maps = unweighted_maps
            else:
                maps = torch.cat((unweighted_maps, maps),dim = 1)

        results = self.ca(maps, atten_maps)
        
        perm_result = results[:nPermMaps]
        result = results[nPermMaps:]
        result = result.view(x.size(0), -1)

        atten_perm_result = atten_results[:nPermMaps]
        atten_result = atten_results[nPermMaps:]
        atten_result = atten_result.view(x.size(0), -1)


        # (nRef*nQuery,)-> (nQuery,nRef)

        perm_result = perm_result.view(perm_x.size(0),-1).transpose(0,1)
        perm_result = perm_result.view(perm_result.size(0),classes, num_support).mean(dim = 2)

        atten_perm_result = atten_perm_result.view(perm_x.size(0),-1).transpose(0,1)
        atten_perm_result = atten_perm_result.view(atten_perm_result.size(0),classes, num_support).mean(dim = 2)
        perm_x = perm_x.view(perm_x.size(0),-1)
        perm_x = perm_x.view(classes, num_support, perm_x.size(1)).mean(dim = 1)
        return result, perm_result, atten_result, atten_perm_result, perm_x, x.view(x.size(0),-1) 
    
    def direct_forward_with_weight(self, x, w, wc=None, attention = False):
        nRef = w.size(0)
        if nRef==self.classes_tr*self.num_support_tr:
            classes = self.classes_tr
            num_support = self.num_support_tr
        else:
            classes = self.classes_val
            num_support = self.num_support_val

        # compute feature
        x = self.f(x)
        w = w.view(classes, num_support, w.size(1)).mean(dim = 1)

        new_feature_maps = channel_wise_mul_per_batch(x,w)
        if attention:
            result = self.a(new_feature_maps)
        else:
            atten_maps = self.a.Attention(new_feature_maps)
            if not self.feature_mode == 'weighted':
                unweighted_new_feature_maps = x.repeat(1,classes,1,1).view((-1,)+x.size()[1:])
                if self.feature_mode == 'unweighted':
                    new_feature_maps = unweighted_new_feature_maps
                else:
                    new_feature_maps = torch.cat((unweighted_new_feature_maps, new_feature_maps),dim = 1)
            if wc is not None:
                assert self.feature_mode == 'weighted'
                new_feature_maps = channel_wise_mul_per_batch(x,wc)
            result = self.ca(new_feature_maps, atten_maps)
        
        return result.view(x.size(0), -1)
    
    def direct_forward_with_perm_weight(self, s, perm_w, perm_wc=None, attention = False):
        # compute feature
        perm_x = self.f(s)
        nRef = s.size(0)
        nQuery = perm_w.size(0)

        if nRef==self.classes_tr*self.num_support_tr:
            classes = self.classes_tr
            num_support = self.num_support_tr
        else:
            classes = self.classes_val
            num_support = self.num_support_val

        new_feature_maps = channel_wise_mul_per_batch(perm_x,perm_w)
        if attention:
            perm_result = self.a(new_feature_maps)
        else:
            atten_maps = self.a.Attention(new_feature_maps)
            if not self.feature_mode == 'weighted':
                unweighted_new_feature_maps = perm_x.repeat(1,nQuery,1,1).view((-1,)+perm_x.size()[1:])
                if self.feature_mode == 'unweighted':
                    new_feature_maps = unweighted_new_feature_maps
                else:
                    new_feature_maps = torch.cat((unweighted_new_feature_maps, new_feature_maps),dim = 1)
            if perm_wc is not None:
                assert self.feature_mode == 'weighted'
                new_feature_maps = channel_wise_mul_per_batch(perm_x, perm_wc)

            perm_result = self.ca(new_feature_maps, atten_maps)


        perm_result = perm_result.view(perm_x.size(0),-1).transpose(0,1)
        perm_result = perm_result.view(perm_result.size(0),classes, num_support).mean(dim = 2)
        return perm_result
    def get_cam(self, x, w):
        '''
        get class activation maps on query image
        '''
        
        nRef = w.size(0)
        if nRef==self.classes_tr*self.num_support_tr:
            classes = self.classes_tr
            num_support = self.num_support_tr
        else:
            classes = self.classes_val
            num_support = self.num_support_val


        # compute feature
        x = self.f(x)
        w = w.view(classes, num_support, w.size(1)).mean(dim = 1)
        new_feature_maps = channel_wise_mul_per_batch(x,w)
        cams = self.a.Attention(new_feature_maps)

        # additional
        cams_reshape = cams.view(x.size(0),-1)
        batch_maxs, _ = torch.max(cams_reshape, dim=-1, keepdim=True)
        batch_mins, _ = torch.min(cams_reshape, dim=-1, keepdim=True)
        cam_normed = torch.div(cams_reshape-batch_mins, batch_maxs - batch_mins)
        cam_normed = cam_normed.view(cams.size())
        return cam_normed

    def get_perm_cam(self, s, perm_w):
        '''
        get query activation maps on reference image
        '''
        # compute feature
        perm_x = self.f(s)
        nRef = s.size(0)
        nQuery = perm_w.size(0)
        if nRef==self.classes_tr*self.num_support_tr:
            classes = self.classes_tr
            num_support = self.num_support_tr
        else:
            classes = self.classes_val
            num_support = self.num_support_val

        new_feature_maps = channel_wise_mul_per_batch(perm_x,perm_w)
        cams = self.a.Attention(new_feature_maps)
        # additional
        cams_reshape = cams.view(nRef,nQuery,-1).transpose(1,0).contiguous().view(nQuery,-1)
        batch_maxs, _ = torch.max(cams_reshape, dim=-1, keepdim=True)
        batch_mins, _ = torch.min(cams_reshape, dim=-1, keepdim=True)
        cam_normed = torch.div(cams_reshape-batch_mins, batch_maxs - batch_mins)
        cam_normed = cam_normed.view(nQuery,nRef,-1).transpose(1,0).contiguous().view(cams.size())
        return cam_normed
    def get_single_cam(self, query, ref):
        '''
        get ref activation maps on query images
        reflect the part that query in similar to ref
        '''
        x = self.f(query)
        w = self.r(self.f(ref)).mean(dim = 0, keepdim = True)
        new_feature_maps = channel_wise_mul_per_batch(x,w)
        cams = self.a.Attention(new_feature_maps)
        return cams