# coding=utf-8

from cfr_loss import cfr_loss as loss_fn, ensembled_loss as en_loss_fn, proto_loss as p_loss, relation_loss as r_loss
from initializer import init_crfnet,init_protonet, init_log_file, init_seed, init_dataloader, init_optim, init_lr_scheduler,init_relationnet

from parser import get_parser

from tqdm import tqdm
import numpy as np
import torch
import os
default_device = 'cuda:0'
def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None, logger = None):
    '''
    Train the model with the reweighting algorithm
    '''

    device = default_device if torch.cuda.is_available() and opt.cuda else 'cpu'

    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_perm_loss = []
    train_acc = []
    train_inter_class_loss = []
    train_perm_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')

    for epoch in range(opt.epochs):
        logger.info('=== Epoch: {} ==='.format(epoch))
        if opt.switch:
            tr_dataloader.dataset.switch_image_size(224)
            val_dataloader.dataset.switch_image_size(224)
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            s = x[:opt.classes_per_it_tr*opt.num_support_tr]
            # commit here to allow standard input combined into the query batch while training
            x = x[opt.classes_per_it_tr*opt.num_support_tr:]
            x, y, s = x, y.cuda(), s
            s = s.repeat([len([int(gpu_id) for gpu_id in opt.gpu if gpu_id.isdigit()]),1,1,1])
            model_output, perm_output, atten_output, perm_atten_output, weight, perm_weight = model(x, s)

            loss,  acc , perm_loss, perm_acc, inter_class_loss, ensemble_loss \
            = en_loss_fn(model_output,perm_output, perm_weight, weight, y,\
                         class_per_it=opt.classes_per_it_tr ,num_support = opt.num_support_tr)
            
            atten_loss,  atten_acc , atten_perm_loss, atten_perm_acc, atten_inter_class_loss, atten_ensemble_loss \
            = en_loss_fn(atten_output,perm_atten_output, perm_weight, weight, y,\
                         class_per_it=opt.classes_per_it_tr ,num_support = opt.num_support_tr)

            loss = loss + atten_loss
            if opt.use_perm:
                loss = loss + perm_loss + atten_perm_loss
            if opt.use_inter_class:
                loss = loss + 0.001*inter_class_loss + 0.1*ensemble_loss + 0.1*atten_ensemble_loss

            loss.backward()

            optim.step()
            train_loss.append(loss.item())
            
            train_acc.append(acc.item())
            train_perm_loss.append(perm_loss.item())
            train_perm_acc.append(perm_acc.item())
            train_inter_class_loss.append(inter_class_loss.item())
        # print(train_loss)
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_perm_loss = np.mean(train_perm_loss[-opt.iterations:])
        avg_inter_class_loss = np.mean(train_inter_class_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        avg_perm_acc = np.mean(train_perm_acc[-opt.iterations:])
        
        logger.info('Avg Train Loss: {}, Avg Perm Loss:{}, Avg InCl Loss:{}, Avg Train Acc: {}, Perm Acc:{}'.format(avg_loss, avg_perm_loss, avg_inter_class_loss, avg_acc, avg_perm_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        
        model.eval()
        with torch.no_grad():
            eps = 1
            val_iter = iter(val_dataloader)

            for batch in tqdm(val_iter):
                x, y = batch
                s = x[:opt.classes_per_it_val*opt.num_support_val]
                x = x[opt.classes_per_it_val*opt.num_support_val:]
                x, y, s = x.cuda(), y.cuda(), s.cuda()
                s = s.repeat([len([int(gpu_id) for gpu_id in opt.gpu if gpu_id.isdigit()]),1,1,1])
                model_output, perm_output, atten_output, perm_atten_output, weight, perm_weight = model(x, s)
                loss,  acc , perm_loss, perm_acc, inter_class_loss, ensemble_loss \
                = en_loss_fn(model_output,perm_output, perm_weight, weight, y,\
                             class_per_it=opt.classes_per_it_val ,num_support = opt.num_support_val)


                val_loss.append(loss.item())
                val_acc.append(acc.item())
            # import pdb; pdb.set_trace()
            avg_loss = np.mean(val_loss[-len(val_iter):])
            avg_acc = np.mean(val_acc[-len(val_iter):])
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
            logger.info('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
            if avg_acc >= best_acc:
                torch.save(model.module.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = model.module.state_dict()


            torch.save(model.module.state_dict(), os.path.join(opt.experiment_root, 'best_model{}.pth'.format(epoch)))



    torch.save(model.module.state_dict(), last_model_path)


    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc

def main(options, logger):
    '''
    Initialize everything and train
    '''
    logger.info('Algorithm options %s' % options)
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
        
    
    if torch.cuda.is_available() and not options.cuda:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    init_seed(options)
    
    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')

    model = init_crfnet(options)
    logger.info('Model Config')
    logger.info(model)
    
    if options.load:
        logger.info('load old model')
        model_path = os.path.join(options.experiment_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path,map_location=default_device))
    model=torch.nn.DataParallel(model,device_ids=range(len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()]))).cuda()
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler, logger = logger)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    del model
    return best_acc
def train_relation(options, logger):
    '''
    Initialize everything and train
    '''
    logger.info('Algorithm options %s' % options)
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
        
    
    if torch.cuda.is_available() and not options.cuda:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    init_seed(options)
    
    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')

    model = init_relationnet(options)
    logger.info('Model Config')
    logger.info(model)
    
    if options.load:
        logger.info('load old model')
        model_path = os.path.join(options.experiment_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path,map_location=default_device), strict = False)
    model=torch.nn.DataParallel(model,device_ids=range(len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()]))).cuda()
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler, logger = logger)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    del model
    return best_acc

def train_proto(options, logger):
    '''
    Initialize everything and train
    '''
    logger.info('Algorithm options %s' % options)
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
        
    
    if torch.cuda.is_available() and not options.cuda:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    init_seed(options)
    
    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')

    model = init_protonet(options)
    logger.info('Model Config')
    logger.info(model)
    device = default_device if torch.cuda.is_available() and options.cuda else 'cpu'
    if options.load:
        logger.info('load old model')
        model_path = os.path.join(options.experiment_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path,map_location=default_device))
    model=torch.nn.DataParallel(model,device_ids=range(len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()])))
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    best_model_path = os.path.join(options.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(options.experiment_root, 'last_model.pth')

    for epoch in range(options.epochs):
        logger.info('=== Epoch: {} ==='.format(epoch))
        if options.switch:
            tr_dataloader.dataset.switch_image_size()
            val_dataloader.dataset.switch_image_size()
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            ref_output = model_output[:options.classes_per_it_tr*options.num_support_tr].view(options.classes_per_it_tr,options.num_support_tr,-1).mean(dim = 1)
            query_output = model_output[options.classes_per_it_tr*options.num_support_tr:]
            loss,  acc = p_loss(query_output, ref_output,y, class_per_it=options.classes_per_it_tr ,num_support = options.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = np.mean(train_loss[-options.iterations:])
        avg_acc = np.mean(train_acc[-options.iterations:])
        
        logger.info('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        
        model.eval()
        with torch.no_grad():
            eps = 1
            val_iter = iter(val_dataloader)

            for batch in tqdm(val_iter):
                x, y = batch
                x, y = x.to(device),y.to(device)

                model_output = model(x)
                ref_output = model_output[:options.classes_per_it_val*options.num_support_val].view(options.classes_per_it_val,options.num_support_val,-1).mean(dim = 1)
                query_output = model_output[options.classes_per_it_val*options.num_support_val:]
                loss,  acc = p_loss(query_output, ref_output, y, class_per_it=options.classes_per_it_val ,num_support = options.num_support_val)

                val_loss.append(loss.item())
                val_acc.append(acc.item())

            avg_loss = np.mean(val_loss[-len(val_iter):])
            avg_acc = np.mean(val_acc[-len(val_iter):])
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
            logger.info('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
            if avg_acc >= best_acc:
                torch.save(model.module.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = model.module.state_dict()

    torch.save(model.module.state_dict(), last_model_path)

def train_relation_attention(options, logger):
    '''
    Initialize everything and train
    '''
    logger.info('Algorithm options %s' % options)
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)
        
    
    if torch.cuda.is_available() and not options.cuda:
        logger.info("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    init_seed(options)
    
    tr_dataloader = init_dataloader(options, 'train')
    val_dataloader = init_dataloader(options, 'val')

    model = init_relationnet(options)
    model.full_load = False
    logger.info('Model Config')
    logger.info(model)
    device = default_device if torch.cuda.is_available() and options.cuda else 'cpu'
    if options.load:
        logger.info('load old model')
        model_path = os.path.join(options.experiment_root, 'best_model.pth')
        model.load_state_dict(torch.load(model_path,map_location=default_device))
    model=torch.nn.DataParallel(model,device_ids=range(len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()])))
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    best_model_path = os.path.join(options.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(options.experiment_root, 'last_model.pth')

    for epoch in range(options.epochs):
        logger.info('=== Epoch: {} ==='.format(epoch))
        if options.switch:
            tr_dataloader.dataset.switch_image_size()
            val_dataloader.dataset.switch_image_size()
        tr_iter = iter(tr_dataloader)
        model.train()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            s = x[:options.classes_per_it_tr*options.num_support_tr]
            # commit here to allow standard input combined into the query batch while training
            x = x[options.classes_per_it_tr*options.num_support_tr:]
            x, y, s = x, y.cuda(), s
            s = s.repeat([len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()]),1,1,1])
            model_output = model(x, s)
            loss,  acc = r_loss(model_output,y, class_per_it=options.classes_per_it_tr ,num_support = options.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = np.mean(train_loss[-options.iterations:])
        avg_acc = np.mean(train_acc[-options.iterations:])
        
        logger.info('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        lr_scheduler.step()
        if val_dataloader is None:
            continue
        
        model.eval()
        with torch.no_grad():
            eps = 1
            val_iter = iter(val_dataloader)

            for batch in tqdm(val_iter):
                x, y = batch
                s = x[:options.classes_per_it_val*options.num_support_val]
                x = x[options.classes_per_it_val*options.num_support_val:]
                x, y, s = x.cuda(), y.cuda(), s.cuda()
                s = s.repeat([len([int(gpu_id) for gpu_id in options.gpu if gpu_id.isdigit()]),1,1,1])
                model_output = model(x, s)
                loss,  acc = r_loss(model_output, y, class_per_it=options.classes_per_it_val ,num_support = options.num_support_val)

                val_loss.append(loss.item())
                val_acc.append(acc.item())

            avg_loss = np.mean(val_loss[-len(val_iter):])
            avg_acc = np.mean(val_acc[-len(val_iter):])
            postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)
            logger.info('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix))
            if avg_acc >= best_acc:
                torch.save(model.module.state_dict(), best_model_path)
                best_acc = avg_acc
                best_state = model.module.state_dict()

    torch.save(model.module.state_dict(), last_model_path)
if __name__ == '__main__':
    options = get_parser().parse_args()
    options.stage = 'train'
    os.environ["CUDA_VISIBLE_DEVICES"]=options.gpu
    logger = init_log_file(options)
    if options.prototypical:
        train_proto(options, logger)
    elif options.relation:
        # train_relation_attention(options, logger)
        train_relation(options, logger)
    else:
        main(options, logger)