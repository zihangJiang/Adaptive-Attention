# coding=utf-8

from cfr_loss import cfr_loss as loss_fn
from cfr_loss import ensembled_loss as en_loss_fn, test_loss as test_loss_fn, proto_loss as p_loss
from initializer import init_crfnet, init_log_file, init_seed, init_dataloader, init_optim, init_lr_scheduler, init_protonet, init_relationnet
from copy import deepcopy
from parser import get_parser
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import os
import cv2
import torch.nn.functional as F

default_device = 'cuda:0'
def get_transform(options):
    if options.dataset in ['MiniImagenet' ,'CUB' ,'StanfordDog','StanfordCar']:
        if options.visulize:
            transform = transforms.Compose([
                                    transforms.RandomCrop(84, padding=8),
                                    transforms.RandomHorizontalFlip(),
                                    lambda x: np.asarray(x),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]],
                                                         std=[x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]])
                                    ])
        else:
            transform = transforms.Compose([
                                    transforms.RandomCrop(84, padding=8),
                                    transforms.RandomHorizontalFlip(),
                                    lambda x: np.asarray(x),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[x/255.0 for x in [120.39586422,  115.59361427, 104.54012653]],
                                                         std=[x/255.0 for x in [70.68188272,  68.27635443,  72.54505529]])
                                    ])

        inverse_t = transforms.Compose([
                                transforms.Normalize(mean=[-x for x in [120.39586422/70.68188272,  115.59361427/68.27635443, 104.54012653/72.54505529]],
                                    std=[255.0/x for x in [70.68188272,  68.27635443,  72.54505529]]),
                                transforms.ToPILImage()
                                ])
    elif options.dataset == 'Omniglot':
        transform = transforms.Compose([
                                transforms.Resize(28),
                                transforms.Pad(4,fill = 255),
                                transforms.RandomCrop(28),
                                transforms.ToTensor(),
                                lambda x: 1-x
                                ])
        inverse_t = transforms.Compose([
                                lambda x: 1-x,
                                transforms.ToPILImage()
                                ])
    return transform, inverse_t

def fine_tune(opt, logger, x, y, model, optim, transform, inverse_t):
    device = default_device if torch.cuda.is_available() and opt.cuda else 'cpu'

    model.train()
    lrs = [1,1,1,0.5]+[0.5 for i in range(opt.epochs)]
    y = y.to(device)
    for epoch in range(opt.epochs):
        optim.zero_grad()
        s = deepcopy(x)
        xx = deepcopy(x)
        for i in range(s.size(0)):
            s[i] = transform(inverse_t(s[i]))
            xx[i] = transform(inverse_t(xx[i]))
        s,xx = s.to(device), xx.to(device)
        #s = x.repeat([len([int(gpu_id) for gpu_id in opt.gpu if gpu_id.isdigit()]),1,1,1])
        model_output, perm_output, atten_output, perm_atten_output, weight, perm_weight = model(xx, s)

        loss,  acc , perm_loss, perm_acc, inter_class_loss, ensemble_loss \
        = en_loss_fn(model_output,perm_output, perm_weight, weight, y,\
                     class_per_it=opt.classes_per_it_tr ,num_support = opt.num_support_tr)
        
        atten_loss,  atten_acc , atten_perm_loss, atten_perm_acc, atten_inter_class_loss, atten_ensemble_loss \
        = en_loss_fn(atten_output,perm_atten_output, perm_weight, weight, y,\
                     class_per_it=opt.classes_per_it_tr ,num_support = opt.num_support_tr)
        loss = loss + atten_loss
        if opt.use_perm:
            loss = loss + perm_loss + atten_perm_loss# no uic = no ensemble
        if opt.use_inter_class:
            loss = loss + 0.001*inter_class_loss + 0.1*ensemble_loss + 0.1*atten_ensemble_loss
        
        loss.backward()
        optim.param_groups[0]['lr'] = opt.learning_rate*lrs[epoch]
        optim.step()
    return model.module.state_dict()
def apply_mask(mask, image):
    width, height = image.shape[0], image.shape[1]
    # mask = mask-np.min(mask)
    # mask = mask/np.max(mask)

    mask = np.reshape(mask,(mask.shape[0], mask.shape[1],1))
    mask = cv2.resize(mask, (width, height))
    a = 0#.25
    mask = (1/(1-a)) *(mask-a)*((mask-a)>0)
    mask = np.uint8(255*mask)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

    result = heatmap*0.3+image*0.5
    # result = heatmap*0.9
    return result
def vis(options):
    for i in range(options.batch_size):
        image = cv2.imread('../exp/visulize/query/{}.png'.format(i))
        for j in range(options.classes_per_it_val):
            perm_image = cv2.imread('../exp/visulize/reference/{}.png'.format(j))
            mask = np.load('../exp/visulize/mask/{}_{}.npy'.format(i,j))
            result = apply_mask(mask, image)
            cv2.imwrite('../exp/visulize/result/{}_map_{}.jpg'.format(i,j), result)
            if not options.relation:
                perm_mask = np.load('../exp/visulize/perm_mask/{}_{}.npy'.format(i,j))
                perm_result = apply_mask(perm_mask, perm_image)
                cv2.imwrite('../exp/visulize/perm_result/{}_map_{}.jpg'.format(j,i), perm_result)


def simple_test(options, logger, strict = False):
    logger.info('Algorithm options %s' % options)
    assert os.path.exists(options.experiment_root)
        
        
    
    if torch.cuda.is_available() and not options.cuda:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    init_seed(options)
    
    test_dataloader = init_dataloader(options, 'test')


    model = init_crfnet(options)
    logger.info('Model Config')
    logger.info(model)
    

    model_path = os.path.join(options.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path,map_location=default_device))
    model=torch.nn.DataParallel(model,device_ids=range(len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()])))
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    device = default_device if torch.cuda.is_available() and options.cuda else 'cpu'
    model.eval()
    val_loss = []
    val_acc = []
    perm_val_loss = []
    perm_val_acc = []
    ensemble_val_loss = []
    ensemble_val_acc = []
    with torch.no_grad():
        test_iter = iter(test_dataloader)
        for batch in tqdm(test_iter):
            x, y = batch
            s = x[:options.classes_per_it_val*options.num_support_val]
            x = x[options.classes_per_it_val*options.num_support_val:]
            x, y, s = x.to(device), y.to(device), s.to(device)
            s = s.repeat([len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()]),1,1,1])
            model_output, perm_output, atten_output, perm_atten_output, weight, perm_weight = model(x, s)
            loss,  acc , perm_loss, perm_acc, ensemble_loss, ensemble_acc \
            = test_loss_fn(model_output,perm_output, perm_weight, weight, y,\
                         class_per_it=options.classes_per_it_val ,num_support = options.num_support_val)
            
            

            val_loss.append(loss.item())
            val_acc.append(acc.item())
            perm_val_loss.append(perm_loss.item())
            perm_val_acc.append(perm_acc.item())
            ensemble_val_loss.append(ensemble_loss.item())
            ensemble_val_acc.append(ensemble_acc.item())
        avg_loss = np.mean(val_loss)
        avg_acc = np.mean(val_acc)
        import pdb; pdb.set_trace()
        perm_avg_loss = np.mean(perm_val_loss)
        perm_avg_acc = np.mean(perm_val_acc)
        ensemble_avg_loss = np.mean(ensemble_val_loss)
        ensemble_avg_acc = np.mean(ensemble_val_acc)

        logger.info('Avg Val Loss: {}, Avg Val Acc: {}, Avg Perm Loss: {}, Avg Perm Acc: {}, Avg Ense Loss: {}, Avg Ense Acc: {}'
            .format(avg_loss, avg_acc, perm_avg_loss, perm_avg_acc, ensemble_avg_loss, ensemble_avg_acc))
def test_proto(options, logger, strict = False):
    '''
    Initialize everything and test
    '''
    logger.info('Algorithm options %s' % options)
    init_seed(options)
    test_dataloader = init_dataloader(options, 'test')
    model = init_protonet(options)
    # logger.info('Model Config')
    # logger.info(model)
    model_path = os.path.join(options.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path,map_location=default_device), strict = strict)
    model=torch.nn.DataParallel(model,device_ids=range(len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()])))
    model.eval()
    device = default_device if torch.cuda.is_available() and options.cuda else 'cpu'
    pred_acc = list()
    transform, inverse_t = get_transform(options)
    test_iter = iter(test_dataloader)

    for batch in tqdm(test_iter):
        pred_better = 0
        x, y = batch
        with torch.no_grad():
            model.eval()
            if options.test_augmentation:
                x1 = deepcopy(x)
                x2 = deepcopy(x)
                x3 = deepcopy(x)
                x4 = deepcopy(x)

                for i in range(x.size(0)):
                    x1[i] = transform(inverse_t(x1[i]))
                    x2[i] = transform(inverse_t(x2[i]))
                    x3[i] = transform(inverse_t(x3[i]))
                    x4[i] = transform(inverse_t(x4[i]))
                x, x1, x2, x3, x4, y = x.to(device), x1.to(device), x2.to(device), x3.to(device), x4.to(device), y.to(device)
                prototype=(model(x)+model(x1)+model(x2)+model(x3)+model(x4))/5
            else:
                x, y = x.to(device), y.to(device)
                prototype=model(x)

            ref_output = prototype[:options.classes_per_it_val*options.num_support_val].view(options.classes_per_it_val,options.num_support_val,-1).mean(dim = 1)
            query_output = prototype[options.classes_per_it_val*options.num_support_val:]
            loss_pred,  acc_pred = p_loss(query_output, ref_output, y, class_per_it=options.classes_per_it_val ,num_support = options.num_support_val)

            pred_acc.append(acc_pred.item())
    ori_avg_acc = np.mean(pred_acc)
    logger.info('Acc: {}'.format(ori_avg_acc))
    ori_std_acc = np.std(pred_acc)
    logger.info('std: {}'.format(ori_std_acc))

def test_relation_attention(options, logger, strict = False):
    '''
    Initialize everything and test
    '''
    logger.info('Algorithm options %s' % options)
    init_seed(options)
    test_dataloader = init_dataloader(options, 'test')
    if options.visulize:
        assert options.iterations<10
        assert options.batch_size<50
        assert options.regular==False
        for name in ['query', 'reference', 'mask', 'result']:
            vis_dir = '../exp/visulize/'+name
            if (not os.path.isdir(vis_dir)):
                os.makedirs(vis_dir)
        # save image of 312*312 for visualization
        # test_dataloader.dataset.switch_image_size(224)
        # test_iter = iter(test_dataloader)
        # transform, inverse_t = get_transform(options)
        # for batch in test_iter:
        #     x, y = batch
        #     s = x[:options.classes_per_it_val*options.num_support_val]
        #     x = x[options.classes_per_it_val*options.num_support_val:]
        #     for i in range(s.size(0)):
        #         im = inverse_t(s[i].cpu())
        #         im.save('../exp/visulize/reference/{}.png'.format(i))
        #     for i in range(x.size(0)):
        #         im = inverse_t(x[i].cpu())
        #         im.save('../exp/visulize/query/{}.png'.format(i))
        # # back to 84*84 setting
        # init_seed(options)
        # test_dataloader = init_dataloader(options, 'test') 
        # test_dataloader.dataset.switch_image_size(224)
    model = init_relationnet(options)
    # logger.info('Model Config')
    # logger.info(model)
    model.full_load = False
    model_path = os.path.join(options.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path,map_location=default_device), strict = strict)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)



    model=torch.nn.DataParallel(model,device_ids=range(len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()])))
    model.eval()
    device = default_device if torch.cuda.is_available() and options.cuda else 'cpu'
    perm_acc = list()
    pred_acc = list()
    combined_acc = list()
    both_acc = list()
    ensemble_acc = list()
    transform, inverse_t = get_transform(options)
    from cfr_loss import label_to_index

    for epoch in range(1):
        if options.switch:
            test_dataloader.dataset.switch_image_size(224)
        test_iter = iter(test_dataloader)

        for batch in tqdm(test_iter):

            x, y = batch
            s = x[:options.classes_per_it_val*options.num_support_val]
            x = x[options.classes_per_it_val*options.num_support_val:]

            with torch.no_grad():
                model.eval()

                x, s, y = x.to(device), s.to(device), y.to(device)

                model_output = model(x, s)
                classes = y[:options.classes_per_it_val\
                            *options.num_support_val].view(options.classes_per_it_val\
                            ,options.num_support_val)[:,0]
                label = y[options.classes_per_it_val*options.num_support_val:]
                target_idx = label_to_index(classes, label)

                if options.visulize:
                    cams = model.module.get_cam(x, s)

                    cams = cams.view(-1,options.classes_per_it_val,cams.size(2),cams.size(3))
                    for i in range(cams.size(0)):
                        idx = torch.argmax(model_output[i])
                        print('example: {},prediction:{}, label: {}'.format(i,idx.item(), target_idx[i].item()))
                        for j in range(5):
                            np.save('../exp/visulize/mask/{}_{}'.format(i, j), cams[i][j].cpu().numpy())



                model.module.load_state_dict(torch.load(model_path,map_location=default_device), strict = strict)

                idx_pred = torch.argmax(model_output, dim = 1)
                acc_pred = idx_pred.eq(target_idx).float().mean()

                pred_acc.append(acc_pred)


    ori_avg_acc = np.mean(pred_acc)
    logger.info('original Acc: {}'.format(ori_avg_acc))
    ori_std_acc = np.std(pred_acc)
    logger.info('Ensemble std: {}'.format(ori_std_acc))
def test_relation(options, logger, strict = False):
    '''
    Initialize everything and test
    '''
    logger.info('Algorithm options %s' % options)
    init_seed(options)
    test_dataloader = init_dataloader(options, 'test')
    if options.visulize:
        assert options.iterations<10
        assert options.batch_size<50
        assert options.regular==False
        for name in ['query', 'reference', 'mask', 'perm_mask', 'result', 'perm_result']:
            vis_dir = '../exp/visulize/'+name
            if (not os.path.isdir(vis_dir)):
                os.makedirs(vis_dir)
        # # save image of 312*312 for visualization
        # test_dataloader.dataset.switch_image_size(224)
        # test_iter = iter(test_dataloader)
        # transform, inverse_t = get_transform(options)
        # for batch in test_iter:
        #     x, y = batch
        #     s = x[:options.classes_per_it_val*options.num_support_val]
        #     x = x[options.classes_per_it_val*options.num_support_val:]
        #     for i in range(s.size(0)):
        #         im = inverse_t(s[i].cpu())
        #         im.save('../exp/visulize/reference/{}.png'.format(i))
        #     for i in range(x.size(0)):
        #         im = inverse_t(x[i].cpu())
        #         im.save('../exp/visulize/query/{}.png'.format(i))
        # # back to 84*84 setting
        # init_seed(options)
        # test_dataloader = init_dataloader(options, 'test') 
        # test_dataloader.dataset.switch_image_size(224)


    model = init_relationnet(options)
    # logger.info('Model Config')
    # logger.info(model)
    model_path = os.path.join(options.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path,map_location=default_device), strict = strict)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)



    model=torch.nn.DataParallel(model,device_ids=range(len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()])))
    model.eval()
    device = default_device if torch.cuda.is_available() and options.cuda else 'cpu'
    perm_acc = list()
    pred_acc = list()
    combined_acc = list()
    both_acc = list()
    ensemble_acc = list()
    transform, inverse_t = get_transform(options)
    from cfr_loss import label_to_index

    for epoch in range(1):
        if options.switch:
            test_dataloader.dataset.switch_image_size(224)
        test_iter = iter(test_dataloader)
        for batch in tqdm(test_iter):
            pred_better = 0
            perm_better = 0
            ensemble = 0
            combined = 0
            both = 0
            x, y = batch
            s = x[:options.classes_per_it_val*options.num_support_val]
            x = x[options.classes_per_it_val*options.num_support_val:]
            x, s, y = x.to(device), s.to(device), y.to(device)
            with torch.no_grad():
                model.eval()
                model_output, perm_output, atten_output, perm_atten_output, weight, perm_weight = model(x, s)

                classes = y[:options.classes_per_it_val\
                            *options.num_support_val].view(options.classes_per_it_val\
                            ,options.num_support_val)[:,0]
                label = y[options.classes_per_it_val*options.num_support_val:]
                target_idx = label_to_index(classes, label)

                if options.visulize:
                    cams = model.module.get_cam(x, s)

                    cams = cams.view(-1,options.classes_per_it_val,cams.size(2),cams.size(3))
                    for i in range(cams.size(0)):
                        idx = torch.argmax(model_output[i])
                        print('example: {},prediction:{}, label: {}'.format(i,idx.item(), target_idx[i].item()))
                        for j in range(5):
                            np.save('../exp/visulize/mask/{}_{}'.format(i, j), cams[i][j].cpu().numpy())

                idx_perm = torch.argmax(perm_output, dim = 1)
                idx_pred = torch.argmax(model_output, dim = 1)
                # idx_ense = torch.argmax(model_output + perm_output, dim = 1)
                # idx_ense = torch.argmax(F.softmax(model_output+perm_output,dim = 1)
                #  + F.softmax(atten_output+atten_perm_output, dim = 1)
                #  , dim = 1)
                idx_ense = torch.argmax(atten_output, dim = 1)
                acc_perm = idx_perm.eq(target_idx).float().mean()
                acc_pred = idx_pred.eq(target_idx).float().mean()
                acc_ense = idx_ense.eq(target_idx).float().mean()
                acc_both = (idx_pred.eq(target_idx)*idx_perm.eq(target_idx)).float().mean()

                perm_acc.append(acc_perm)
                pred_acc.append(acc_pred)
                ensemble_acc.append(acc_ense)
                both_acc.append(acc_both)


    both_avg_acc = np.mean(both_acc)
    logger.info('Both right Acc: {}'.format(both_avg_acc))
    ori_avg_acc = np.mean(pred_acc)
    logger.info('original Acc: {}'.format(ori_avg_acc))
    perm_avg_acc = np.mean(perm_acc)
    logger.info('Perm Acc: {}'.format(perm_avg_acc))
    # combined_avg_acc = np.mean(combined_acc)
    # logger.info('Combined Acc: {}'.format(combined_avg_acc))
    ensemble_avg_acc = np.mean(ensemble_acc)
    ensemble_std_acc = np.std(ensemble_acc)

    logger.info('Ensemble Acc: {}'.format(ensemble_avg_acc))
    logger.info('Ensemble std: {}'.format(ensemble_std_acc))
    logger.info('95% Confidence interval: {}'.format(ensemble_std_acc*1.96/np.sqrt(600)))
def test(options, logger, strict = False):
    '''
    Initialize everything and test
    '''
    logger.info('Algorithm options %s' % options)
    init_seed(options)
    test_dataloader = init_dataloader(options, 'test')
    if options.visulize:
        assert options.iterations<10
        assert options.batch_size<100
        assert options.regular==False
        for name in ['query', 'reference', 'mask', 'perm_mask', 'result', 'perm_result']:
            vis_dir = '../exp/visulize/'+name
            if (not os.path.isdir(vis_dir)):
                os.makedirs(vis_dir)
        # # save image of 312*312 for visualization
        # test_dataloader.dataset.switch_image_size(224)
        # test_iter = iter(test_dataloader)
        # transform, inverse_t = get_transform(options)
        # for batch in test_iter:
        #     x, y = batch
        #     s = x[:options.classes_per_it_val*options.num_support_val]
        #     x = x[options.classes_per_it_val*options.num_support_val:]
        #     for i in range(s.size(0)):
        #         im = inverse_t(s[i].cpu())
        #         im.save('../exp/visulize/reference/{}.png'.format(i))
        #     for i in range(x.size(0)):
        #         im = inverse_t(x[i].cpu())
        #         im.save('../exp/visulize/query/{}.png'.format(i))
        # # back to 84*84 setting
        # init_seed(options)
        # test_dataloader = init_dataloader(options, 'test') 
        # test_dataloader.dataset.switch_image_size(224)


    model = init_crfnet(options)
    # logger.info('Model Config')
    # logger.info(model)
    model_path = os.path.join(options.experiment_root, 'best_model69.pth')
    model.load_state_dict(torch.load(model_path,map_location=default_device), strict = strict)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)

    # torch.save(model.a.state_dict(), os.path.join(options.experiment_root, 'a_model.pth'))
    # torch.save(model.f.state_dict(), os.path.join(options.experiment_root, 'f_model.pth'))

    model=torch.nn.DataParallel(model,device_ids=range(len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()])))
    model.eval()
    device = default_device if torch.cuda.is_available() and options.cuda else 'cpu'
    perm_acc = list()
    pred_acc = list()
    combined_acc = list()
    both_acc = list()
    ensemble_acc = list()
    transform, inverse_t = get_transform(options)
    from cfr_loss import label_to_index

    for epoch in range(1):
        if options.switch:
            test_dataloader.dataset.switch_image_size(224)
        test_iter = iter(test_dataloader)

        for batch in tqdm(test_iter):
            pred_better = 0
            perm_better = 0
            ensemble = 0
            combined = 0
            both = 0
            x, y = batch
            s = x[:options.classes_per_it_val*options.num_support_val]
            x = x[options.classes_per_it_val*options.num_support_val:]


            if options.fine_tune:
                reference = s
                tune_y = y[:options.classes_per_it_val*options.num_support_val]
                

                tune_state = fine_tune(options, logger, reference, tune_y, model, optim, transform, inverse_t)
                model.module.load_state_dict(tune_state)

            with torch.no_grad():
                model.eval()
                if options.double:
                    rc = model.module.rc
                
                ra = model.module.r
                f = model.module.f
                if options.test_augmentation:
                    s1 = deepcopy(s)
                    s2 = deepcopy(s)
                    s3 = deepcopy(s)
                    s4 = deepcopy(s)
                    x1 = deepcopy(x)
                    x2 = deepcopy(x)
                    x3 = deepcopy(x)
                    x4 = deepcopy(x)
                    for i in range(s.size(0)):
                        s1[i] = transform(inverse_t(s1[i]))
                        s2[i] = transform(inverse_t(s2[i]))
                        s3[i] = transform(inverse_t(s3[i]))
                        s4[i] = transform(inverse_t(s4[i]))

                    for i in range(x.size(0)):
                        x1[i] = transform(inverse_t(x1[i]))
                        x2[i] = transform(inverse_t(x2[i]))
                        x3[i] = transform(inverse_t(x3[i]))
                        x4[i] = transform(inverse_t(x4[i]))
                    x, x1, x2, x3, x4, y, s, s1, s2, s3, s4, = x.to(device), x1.to(device), x2.to(device), x3.to(device), x4.to(device), y.to(device), s.to(device), s1.to(device), s2.to(device), s3.to(device), s4.to(device)
                    # attention weight vector
                    perm_weight=(ra(f(x))+ra(f(x1))+ra(f(x2))+ra(f(x3))+ra(f(x4)))/5
                    weight=(ra(f(s))+ra(f(s1))+ra(f(s2))+ra(f(s3))+ra(f(s4)))/5
                    perm_weight = ra.scale_weight * perm_weight/torch.sqrt(torch.sum(perm_weight*perm_weight,1).view(perm_weight.size(0),1))
                    weight = ra.scale_weight * weight/torch.sqrt(torch.sum(weight*weight,1).view(weight.size(0),1))
                    if options.double:
                        # classify weight vector
                        perm_weight_classify=(rc(f(x))+rc(f(x1))+rc(f(x2))+rc(f(x3))+rc(f(x4)))/5
                        weight_classify=(rc(f(s))+rc(f(s1))+rc(f(s2))+rc(f(s3))+rc(f(s4)))/5
                        perm_weight_classify = rc.scale_weight * perm_weight_classify/torch.sqrt(torch.sum(perm_weight_classify*perm_weight_classify,1).view(perm_weight_classify.size(0),1))
                        weight_classify = rc.scale_weight * weight_classify/torch.sqrt(torch.sum(weight_classify*weight_classify,1).view(weight_classify.size(0),1))
                else:
                    x, s, y = x.to(device), s.to(device), y.to(device)
                    # attention weight vector
                    perm_weight=ra(f(x))
                    weight = ra(f(s))
                    if options.double:
                        # classify weight vector
                        perm_weight_classify=rc(f(x))
                        weight_classify = rc(f(s))
                if options.double:
                    model_output = model.module.direct_forward_with_weight(x, weight, weight_classify)
                    atten_output = model.module.direct_forward_with_weight(x, weight,weight_classify,attention = True)
                    perm_output = model.module.direct_forward_with_perm_weight(s, perm_weight, perm_weight_classify)
                    atten_perm_output = model.module.direct_forward_with_perm_weight(s, perm_weight, perm_weight_classify,attention = True)
                else:
                    model_output = model.module.direct_forward_with_weight(x, weight)
                    atten_output = model.module.direct_forward_with_weight(x, weight,attention = True)
                    perm_output = model.module.direct_forward_with_perm_weight(s, perm_weight)
                    atten_perm_output = model.module.direct_forward_with_perm_weight(s, perm_weight,attention = True)

                classes = y[:options.classes_per_it_val\
                            *options.num_support_val].view(options.classes_per_it_val\
                            ,options.num_support_val)[:,0]
                label = y[options.classes_per_it_val*options.num_support_val:]
                target_idx = label_to_index(classes, label)

                if options.visulize:
                    cams = model.module.get_cam(x, weight)
                    perm_cams = model.module.get_perm_cam(s, perm_weight)

                    cams = cams.view(-1,options.classes_per_it_val,cams.size(2),cams.size(3))
                    perm_cams = perm_cams.view(options.classes_per_it_val, -1, perm_cams.size(2),perm_cams.size(3))
                    

                    for i in range(s.size(0)):
                        im = inverse_t(s[i].cpu())
                        im.save('../exp/visulize/reference/{}.png'.format(i))
                    for i in range(cams.size(0)):
                        idx = torch.argmax(F.softmax(model_output[i]+perm_output[i]) + F.softmax(atten_output[i]+atten_perm_output[i]))
                        print('example: {},prediction:{}, label: {}'.format(i,idx.item(), target_idx[i].item()))
                        for j in range(5):
                            np.save('../exp/visulize/mask/{}_{}'.format(i, j), cams[i][j].cpu().numpy())
                            np.save('../exp/visulize/perm_mask/{}_{}'.format(i,j), perm_cams[j][i].cpu().numpy())
                        im = inverse_t(x[i].cpu())
                        im.save('../exp/visulize/query/{}.png'.format(i))


                model.module.load_state_dict(torch.load(model_path,map_location=default_device), strict = strict)

                idx_perm = torch.argmax(perm_output, dim = 1)
                idx_pred = torch.argmax(model_output, dim = 1)
                idx_ense = torch.argmax(model_output + perm_output, dim = 1)
                # idx_ense = torch.argmax(F.softmax(model_output+perm_output,dim = 1)
                #  + F.softmax(atten_output+atten_perm_output, dim = 1)
                #  , dim = 1)
                acc_perm = idx_perm.eq(target_idx).float().mean()
                acc_pred = idx_pred.eq(target_idx).float().mean()
                acc_ense = idx_ense.eq(target_idx).float().mean()
                acc_both = (idx_pred.eq(target_idx)*idx_perm.eq(target_idx)).float().mean()

                perm_acc.append(acc_perm)
                pred_acc.append(acc_pred)
                ensemble_acc.append(acc_ense)
                both_acc.append(acc_both)


    both_avg_acc = np.mean(both_acc)
    logger.info('Both right Acc: {}'.format(both_avg_acc))
    ori_avg_acc = np.mean(pred_acc)
    logger.info('original Acc: {}'.format(ori_avg_acc))
    perm_avg_acc = np.mean(perm_acc)
    logger.info('Perm Acc: {}'.format(perm_avg_acc))
    # combined_avg_acc = np.mean(combined_acc)
    # logger.info('Combined Acc: {}'.format(combined_avg_acc))
    ensemble_avg_acc = np.mean(ensemble_acc)
    ensemble_std_acc = np.std(ensemble_acc)

    logger.info('Ensemble Acc: {}'.format(ensemble_avg_acc))
    logger.info('Ensemble std: {}'.format(ensemble_std_acc))
    logger.info('95% Confidence interval: {}'.format(ensemble_std_acc*1.96/np.sqrt(600)))

if __name__ == '__main__':
    options = get_parser().parse_args()
    options.stage = 'test'
    os.environ["CUDA_VISIBLE_DEVICES"]=options.gpu
    logger = init_log_file(options, prefix='TEST')
    if options.prototypical:
        test_proto(options, logger)
    elif options.relation:
        test_relation(options, logger)
    else:
        test(options, logger, False)
    if options.visulize:
        vis(options)
