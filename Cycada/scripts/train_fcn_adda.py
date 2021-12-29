import logging
import os
import os.path
from argparse import ArgumentParser
from collections import deque
import itertools
from datetime import datetime

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from tensorboardX import SummaryWriter

from PIL import Image
from torch.autograd import Variable

from Cycada.cycada.data.adda_datasets import AddaDataLoader
from Cycada.cycada.models import get_model
from Cycada.cycada.models import models
from Cycada.cycada.models import Discriminator
from Cycada.cycada.util import config_logging
from Cycada.cycada.util import to_tensor_raw
from Cycada.cycada.tools.util import make_variable

import os
import sys
import torch
import time
import warnings

import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from torch.utils.data import ConcatDataset
from argparse import ArgumentParser
from baseline.builders.model_builder import build_model
from baseline.builders.validation_builder import predict_multiscale_sliding
from baseline.utils.utils import setup_seed, netParams
from baseline.utils.plot_log import draw_log
from baseline.utils.record_log import record_log
from baseline.utils.earlyStopping import EarlyStopping
from baseline.utils.losses.loss import CrossEntropyLoss2d
from torch.optim.lr_scheduler import MultiStepLR
from baseline.model_data_loader import SenmanticData


def forward_pass(net, discriminator, im, requires_grad=False, discrim_feat=False):
    if discrim_feat:
        score, feat = net(im)
        dis_score = discriminator(feat)
    else:
        score = net(im)
        dis_score = discriminator(score)
    if not requires_grad:
        score = Variable(score.data, requires_grad=False)
        
    return score, dis_score


def supervised_loss(score, label, weights=None):
    loss_fn_ = torch.nn.NLLLoss(weight=weights, reduction='mean', ignore_index=255)
    loss = loss_fn_(F.log_softmax(score, dim=1), label)
    return loss


def discriminator_loss(score, target_val, lsgan=False):
    if lsgan:
        loss = 0.5 * torch.mean((score - target_val)**2)
    else:
        _,_,h,w = score.size()
        target_val_vec = Variable(target_val * torch.ones(1,h,w),requires_grad=False).long().cuda()
        loss = supervised_loss(score, target_val_vec)
    return loss

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n,n)

def seg_accuracy(score, label, num_cls):
    _, preds = torch.max(score.data, 1)
    hist = fast_hist(label.cpu().numpy().flatten(),
            preds.cpu().numpy().flatten(), num_cls)
    intersections = np.diag(hist)
    unions = (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-8) * 100
    acc = np.diag(hist).sum() / hist.sum()
    return intersections, unions, acc


def main(args):

    # So data is sampled in consistent way
    np.random.seed(1337)
    torch.manual_seed(1337)
    logdir = 'runs/{:s}_{:s}/{:s}_to_{:s}/lr{:.1g}_ld{:.2g}_lg{:.2g}'.format(args.model, args.dataset, args.datafolder[0],
            args.datafolder[1], args.lr, args.lambda_d, args.lambda_g)
    if args.weights_shared:
        logdir += '_weightshared'
    else:
        logdir += '_weightsunshared'
    if args.discrim_feat:
        logdir += '_discrimfeat'
    else:
        logdir += '_discrimscore'
    logdir += '/' + datetime.now().strftime('%Y_%b_%d-%H:%M')
    writer = SummaryWriter(log_dir=logdir)


    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    config_logging()
    print('Train Discrim Only', args.train_discrim_only)
    # net = get_model(args.model, pretrained=True, weights_init=args.weights_init, output_last_ft=args.discrim_feat)
    net = get_model(args)
    if args.weights_shared:
        net_src = net # shared weights
    # else:
    #     net_src = get_model(args)
    #     net_src.eval()

    odim = 1 if args.lsgan else 2
    idim = args.num_cls if not args.discrim_feat else 3328
    print('discrim_feat', args.discrim_feat, idim)
    print('discriminator init weights: ', args.weights_discrim)
    discriminator = Discriminator(input_dim=idim, output_dim=odim, 
            pretrained=not (args.weights_discrim==None),
            weights_init=args.weights_discrim).cuda()

    loader = AddaDataLoader(args)
    print('datafolder', args.datafolder)
  
    # setup optimizers
    opt_dis = torch.optim.SGD(discriminator.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=0.0005)
    opt_rep = torch.optim.SGD(net.parameters(), lr=args.lr,
            momentum=args.momentum, weight_decay=0.0005)

    iteration = 0
    num_update_g = 0
    last_update_g = -1
    losses_super_s = deque(maxlen=100)
    losses_super_t = deque(maxlen=100)
    losses_dis = deque(maxlen=100)
    losses_rep = deque(maxlen=100)
    accuracies_dom = deque(maxlen=100)
    intersections = np.zeros([100,args.num_cls])
    unions = np.zeros([100, args.num_cls])
    accuracy = deque(maxlen=100)
    print('max iter:', args.max_iter)
   
    net.train()
    discriminator.train()

    while iteration < args.max_iter:
        
        for im_s, im_t, label_s, label_t in loader:
            
            if iteration > args.max_iter:
                break
           
            info_str = 'Iteration {}: '.format(iteration)
            
            ###########################
            # 1. Setup Data Variables #
            ###########################
            im_s = make_variable(im_s, requires_grad=False)
            label_s = make_variable(label_s, requires_grad=False)
            im_t = make_variable(im_t, requires_grad=False)
            label_t = make_variable(label_t, requires_grad=False)
           
            #############################
            # 2. Optimize Discriminator #
            #############################
            
            # zero gradients for optimizer
            opt_dis.zero_grad()
            opt_rep.zero_grad()
            
            # extract features
            if args.discrim_feat:
                score_s, feat_s = net_src(im_s)
                score_s = Variable(score_s.data, requires_grad=False)
                f_s = Variable(feat_s.data, requires_grad=False)
            else:
                score_s = Variable(net_src(im_s).data, requires_grad=False)
                f_s = score_s
            dis_score_s = discriminator(f_s)
            
            if args.discrim_feat:
                score_t, feat_t = net(im_t)
                score_t = Variable(score_t.data, requires_grad=False)
                f_t = Variable(feat_t.data, requires_grad=False)
            else:
                score_t = Variable(net(im_t).data, requires_grad=False)
                f_t = score_t
            dis_score_t = discriminator(f_t)
            
            dis_pred_concat = torch.cat((dis_score_s, dis_score_t))

            # prepare real and fake labels
            batch_t,_,h,w = dis_score_t.size()
            batch_s,_,_,_ = dis_score_s.size()
            dis_label_concat = make_variable(
                    torch.cat(
                        [torch.ones(batch_s,h,w).long(), 
                        torch.zeros(batch_t,h,w).long()]
                        ), requires_grad=False)

            # compute loss for discriminator
            loss_dis = supervised_loss(dis_pred_concat, dis_label_concat)
            (args.lambda_d * loss_dis).backward()
            losses_dis.append(loss_dis.item())

            # optimize discriminator
            opt_dis.step()

            # compute discriminator acc
            pred_dis = torch.squeeze(dis_pred_concat.max(1)[1])
            dom_acc = (pred_dis == dis_label_concat).float().mean().item() 
            accuracies_dom.append(dom_acc * 100.)

            # add discriminator info to log
            info_str += " domacc:{:0.1f}  D:{:.3f}".format(np.mean(accuracies_dom), 
                    np.mean(losses_dis))
            writer.add_scalar('loss/discriminator', np.mean(losses_dis), iteration)
            writer.add_scalar('acc/discriminator', np.mean(accuracies_dom), iteration)

            ###########################
            # Optimize Target Network #
            ###########################
           
            dom_acc_thresh = 60

            if not args.train_discrim_only and np.mean(accuracies_dom) > dom_acc_thresh:
              
                last_update_g = iteration
                num_update_g += 1 
                if num_update_g % 1 == 0:
                    print('Updating G with adversarial loss ({:d} times)'.format(num_update_g))

                # zero out optimizer gradients
                opt_dis.zero_grad()
                opt_rep.zero_grad()

                # extract features
                if args.discrim_feat:
                    score_t, feat_t = net(im_t)
                    score_t = Variable(score_t.data, requires_grad=False)
                    f_t = feat_t 
                else:
                    score_t = net(im_t)
                    f_t = score_t

                #score_t = net(im_t)
                dis_score_t = discriminator(f_t)

                # create fake label
                batch,_,h,w = dis_score_t.size()
                target_dom_fake_t = make_variable(torch.ones(batch,h,w).long(), requires_grad=False)

                # compute loss for target net
                loss_gan_t = supervised_loss(dis_score_t, target_dom_fake_t)
                (args.lambda_g * loss_gan_t).backward()
                losses_rep.append(loss_gan_t.item())
                writer.add_scalar('loss/generator', np.mean(losses_rep), iteration)
                
                # optimize target net
                opt_rep.step()

                # log net update info
                info_str += ' G:{:.3f}'.format(np.mean(losses_rep))
               
            if (not args.train_discrim_only) and args.weights_shared and (np.mean(accuracies_dom) > dom_acc_thresh):
               
                print('Updating G using source supervised loss.')

                # zero out optimizer gradients
                opt_dis.zero_grad()
                opt_rep.zero_grad()

                # extract features
                if args.discrim_feat:
                    score_s, _ = net(im_s)
                else:
                    score_s = net(im_s)

                loss_supervised_s = supervised_loss(score_s, label_s)
                loss_supervised_s.backward()
                losses_super_s.append(loss_supervised_s.item())
                info_str += ' clsS:{:.2f}'.format(np.mean(losses_super_s))
                writer.add_scalar('loss/supervised/source', np.mean(losses_super_s), iteration)

                # optimize target net
                opt_rep.step()

            # compute supervised losses for target -- monitoring only!!!
            loss_supervised_t = supervised_loss(score_t, label_t)
            losses_super_t.append(loss_supervised_t.item())
            info_str += ' clsT:{:.2f}'.format(np.mean(losses_super_t))
            writer.add_scalar('loss/supervised/target', np.mean(losses_super_t), iteration)

            ###########################
            # Log and compute metrics #
            ###########################
            if iteration % 10 == 0 and iteration > 0:
                
                # compute metrics
                intersection,union,acc = seg_accuracy(score_t, label_t.data, args.num_cls)
                intersections = np.vstack([intersections[1:,:], intersection[np.newaxis,:]])
                unions = np.vstack([unions[1:,:], union[np.newaxis,:]]) 
                accuracy.append(acc.item() * 100)
                acc = np.mean(accuracy)
                mIoU =  np.mean(np.maximum(intersections, 1) / np.maximum(unions, 1)) * 100
              
                info_str += ' acc:{:0.2f}  mIoU:{:0.2f}'.format(acc, mIoU)
                writer.add_scalar('metrics/acc', np.mean(accuracy), iteration)
                writer.add_scalar('metrics/mIoU', np.mean(mIoU), iteration)
                logging.info(info_str)
                  
            iteration += 1

            ################
            # Save outputs #
            ################

            # every 500 iters save current model
            if iteration % 500 == 0:
                os.makedirs(args.output, exist_ok=True)
                if not args.train_discrim_only:
                    torch.save(net.state_dict(),
                            '{}/net-itercurr.pth'.format(args.output))
                torch.save(discriminator.state_dict(),
                        '{}/discriminator-itercurr.pth'.format(args.output))

            # save labeled snapshots
            if iteration % args.snapshot == 0:
                os.makedirs(args.output, exist_ok=True)
                if not args.train_discrim_only:
                    torch.save(net.state_dict(),
                            '{}/net-iter{}.pth'.format(args.output, iteration))
                torch.save(discriminator.state_dict(),
                        '{}/discriminator-iter{}.pth'.format(args.output, iteration))

            # if iteration - last_update_g >= len(loader):
            #     print('No suitable discriminator found -- returning.')
            #     torch.save(net.state_dict(),
            #             '{}/net-iter{}.pth'.format(args.output, iteration))
            #     iteration = args.max_iter # make sure outside loop breaks
            #     break

    writer.close()


def parse_args():
    parser = ArgumentParser(description='Semantic segmentation with pytorch')
    # model and dataset

    parser.add_argument('--dataset', type=str, default="EPFL", choices=['EPFL', 'comballaz'],
                        help="dataset to train: EPFL or comballaz")
    parser.add_argument('--datafolder', type=str, default=['train_drone_sim', 'train_drone_real'],
                        help="dataset folder name for sim and real data")
    parser.add_argument('--rootdir', default="/DataDisk/visnav_semantics/baseline/datasets",
                        help="Directory where the dataset are in")
    parser.add_argument('--lr', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--fs_model', type=str, default='/media/shanci/Samsung_T5/checkpoint/EPFL'
                                                        '/with_real_data_inplace/best_model.pth',
                        help='path to the pretrained semantic segmentation model in synthetic field')
    parser.add_argument('--momentum', type=float, default=0.9, help="initial momentum")
    parser.add_argument('--batch_size', type=int, default=2, help="the batch size")
    parser.add_argument('--augmentation', type=bool, default=True, help="whether augment the input image")
    parser.add_argument('--snapshot', type=float, default=5000, help="snapshot")
    parser.add_argument('--downscale', type=int, default=None, help="downscale")
    parser.add_argument('--crop_size', type=int, default=None, help="crop_size")
    parser.add_argument('--half_crop', type=bool, default=False, help="augmentation whether perform half crop")
    parser.add_argument('--shuffle', type=bool, default=False, help="whether shuffle the input data")
    parser.add_argument('--weights_discrim', default=None,
                        help="whether use real data in training")
    parser.add_argument('--weights_init', default='/media/shanci/Samsung_T5/cyclegta/drn26-test_drone_sim-iter115000.pth',
                        help="whether use real data in training")
    parser.add_argument('--model', type=str, default="debug", help="model name")
    parser.add_argument('--lsgan', type=bool, default=False)
    parser.add_argument('--num_cls', type=int, default=6, help="number of classes")
    parser.add_argument('--num_workers', type=int, default=8, help="number of workers to load data")
    parser.add_argument('--gpu', default="0", help="default GPU devices 0")
    parser.add_argument('--max_iter', type=int, default=13500*50,
                        help="the number of iter for train")
    parser.add_argument('--lambda_d', type=float, default=1.0)
    parser.add_argument('--lambda_g', type=float, default=1.0)
    parser.add_argument('--train_discrim_only', type=bool, default=False)
    parser.add_argument('--discrim_feat', type=bool, default=True)
    parser.add_argument('--weights_shared', type=bool, default=True)
    args = parser.parse_args()

    if args.weights_shared:
        flag_weights_shared = 'weights_shared'
    else:
        flag_weights_shared = 'weights_unshared'
    if args.discrim_feat:
        flag_discrim_feat = 'discrim_feat'
    else:
        flag_discrim_feat = 'discrim_score'

    resdir = "/DataDisk/FCN_adda/" + str(args.datafolder[0]) + "_to_" + str(args.datafolder[1]) + \
             "/adda_sgd/" + flag_weights_shared +"_nolsgan_" + flag_discrim_feat

    outdir = os.path.join(resdir, args.model, 'lr_'+str(args.lr) + '_ld' + str(args.lambda_d) +
                          '_lg' + str(args.lambda_g) +'_momentum' + str(args.momentum))

    args.output = outdir

    return args



if __name__ == '__main__':
    args = parse_args()
    main(args)

