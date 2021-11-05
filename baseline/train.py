import os
import sys
import torch
import time
import warnings

import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils import data
from argparse import ArgumentParser
from builders.model_builder import build_model
from builders.validation_builder import predict_multiscale_sliding
from baseline.utils.utils import setup_seed, netParams
from baseline.utils.plot_log import draw_log
from baseline.utils.record_log import record_log
from baseline.utils.earlyStopping import EarlyStopping
from baseline.utils.losses.loss import CrossEntropyLoss2d, FocalLoss2d, DiceLoss
from torch.optim.lr_scheduler import MultiStepLR
from model_data_loader import SenmanticData

warnings.filterwarnings('ignore')

sys.setrecursionlimit(1000000)  # solve problem 'maximum recursion depth exceeded'
GLOBAL_SEED = 88


def train(args, train_loader, model, criterion, optimizer, epoch, device):
    """
    args:
       train_loader: loaded for training dataset
       model       : model
       criterion   : loss function
       optimizer   : optimization algorithm, such as ADAM or SGD
       epoch       : epoch number
       device      : cuda
    return: average loss, lr
    """

    model.train()
    epoch_loss = []

    total_batches = len(train_loader)
    pbar = tqdm(iterable=enumerate(train_loader),
                total=total_batches,
                desc='Epoch {}/{}'.format(epoch, args.max_epochs))

    lr = optimizer.param_groups[0]['lr']

    for iteration, batch in pbar:

        optimizer.zero_grad()

        images, labels, _ = batch

        images = images.to(device)
        labels = labels.to(device).long()
        output = model(images)

        if args.loss == 'dice_loss':
            output = F.one_hot(output, args.classes)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)

    # torch.cuda.empty_cache()
    return average_epoch_loss_train, lr


def main(args):
    """
    args:
       args: global arguments
    """
    # set the seed
    setup_seed(GLOBAL_SEED)
    # cudnn.enabled = True
    # cudnn.benchmark = True  # find the optimal configuration
    # cudnn.deterministic = True  # reduce volatility

    # build the model and initialization weights
    model = build_model(args.model, args.classes, args.backbone, args.pretrained, args.out_stride, args.mult_grid)

    # define loss function, respectively

    # Default uses cross quotient loss function
    if args.loss == 'CrossEntropyLoss2d':
        criterion = CrossEntropyLoss2d()
    elif args.loss == 'DiceLoss':
        criterion = DiceLoss()
    elif args.loss == 'FocalLoss2d':
        criterion = FocalLoss2d()

    # load train set
    train_set = SenmanticData('./datasets/' + args.dataset + '/train_sim', normalization=args.normalization)
    val_set = SenmanticData('./datasets/' + args.dataset + '/val_sim', normalization=args.normalization)
    test_set = SenmanticData('./datasets/' + args.dataset + '/test_real', normalization=args.normalization)

    # move model and criterion on cuda
    if args.cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else RuntimeError)
        model = model.to(device)
        trainLoader = data.DataLoader(train_set, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.batch_size, pin_memory=True, drop_last=False)
        valLoader = data.DataLoader(val_set, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.batch_size, pin_memory=True, drop_last=False)
        testLoader = data.DataLoader(test_set, batch_size=args.batch_size,
                                     shuffle=True, num_workers=args.batch_size, pin_memory=True, drop_last=False)

        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    parameters = model.parameters()

    if args.optim == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=5e-4)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=5e-4)

    scheduler = MultiStepLR(optimizer, milestones=[10, 20, 25, 30, 40], gamma=0.5)

    # initial log file val output save
    if args.normalization:
        dir_normal = 'normalized'
    else:
        dir_normal = 'not_normalized'

    args.savedir = os.path.join(args.savedir, args.dataset, args.model, args.loss, dir_normal)
    if not os.path.exists(args.savedir) and args.local_rank == 0:
        os.makedirs(args.savedir)

    # save_seg_dir
    args.save_seg_dir = os.path.join(args.savedir, 'result_figure')
    if not os.path.exists(args.save_seg_dir) and args.local_rank == 0:
        os.makedirs(args.save_seg_dir)

    recorder = record_log(args)

    class_name = ['No data', 'Ground', 'Vegetation', 'Buildings', 'Water', 'Bridges']
    label_index = [0, 1, 2, 3, 4, 5]
    class_dict_df = pd.DataFrame([class_name, label_index], index=['class_name', 'label_index']).T
    class_dict_df = class_dict_df.set_index('label_index')

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=300)
    start_epoch = 1
    if args.local_rank == 0:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
              ">>>>>>>>>>>  beginning training   >>>>>>>>>>>\n"
              ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")

    epoch_list = []
    lossTr_list = []
    Miou_list = []
    lossVal_list = []
    Miou = 0
    Best_Miou = 0
    # continue training
    if args.resume:
        logger, lines = recorder.resume_logfile()
        for index, line in enumerate(lines):
            lossTr_list.append(float(line.strip().split()[2]))
            if len(line.strip().split()) != 3:
                epoch_list.append(int(line.strip().split()[0]))
                lossVal_list.append(float(line.strip().split()[3]))
                Miou_list.append(float(line.strip().split()[5]))

        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cuda')
            start_epoch = checkpoint['epoch'] + 1
            check_list = [i for i in checkpoint['model'].items()]
            # Read weights with multiple cards, and continue training with a single card this time
            if 'module.' in check_list[0][0]:
                new_stat_dict = {}
                for k, v in checkpoint['model'].items():
                    new_stat_dict[k[:]] = v
                model.load_state_dict(new_stat_dict, strict=True)
            # Read the training weight of a single card, and continue training with a single card this time
            else:
                model.load_state_dict(checkpoint['model'])

            optimizer.load_state_dict(checkpoint['optimizer'])

            if args.local_rank == 0:
                print("loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            if args.local_rank == 0:
                print("no checkpoint found at '{}'".format(args.resume))
    else:
        logger = recorder.initial_logfile()
        logger.flush()

    for epoch in range(start_epoch, args.max_epochs + 1):
        start_time = time.time()
        # training
        train_start = time.time()

        lossTr, lr = train(args, trainLoader, model, criterion, optimizer, epoch, device)
        scheduler.step()
        if args.local_rank == 0:
            lossTr_list.append(lossTr)

        # save the checkpoints
        model_file_name = args.savedir + '/checkpoint_epoch_{}.pth'.format(epoch)
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, model_file_name)

        train_end = time.time()
        train_per_epoch_seconds = train_end - train_start
        validation_per_epoch_seconds = 60  # init validation time

        # validation if mode==validation, predict with label; elif mode==predict, predict without label.

        if epoch % args.val_epochs == 0 or epoch == 1 or args.max_epochs - 10 < epoch <= args.max_epochs:
            validation_start = time.time()

            loss, FWIoU, Miou, MIoU, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F1_avg = \
                predict_multiscale_sliding(args=args, model=model,
                                           class_dict_df=class_dict_df,
                                           testLoader=valLoader,
                                           # scales=[1.25, 1.5, 1.75, 2.0],
                                           scales=[1.0],
                                           overlap=0.3,
                                           criterion=criterion,
                                           mode=args.predict_type,
                                           save_result=True)
            torch.cuda.empty_cache()

            if args.local_rank == 0:
                epoch_list.append(epoch)
                Miou_list.append(Miou)
                lossVal_list.append(loss.item())
                # record trainVal information
                recorder.record_trainVal_log(logger, epoch, lr, lossTr, loss,
                                             FWIoU, Miou, MIoU, PerCiou_set, Pa, Mpa,
                                             PerCpa_set, MF, F_set, F1_avg,
                                             class_dict_df)

                torch.cuda.empty_cache()
                validation_end = time.time()
                validation_per_epoch_seconds = validation_end - validation_start
        else:
            if args.local_rank == 0:
                # record train information
                recorder.record_train_log(logger, epoch, lr, lossTr)

        if args.local_rank == 0:
            # draw log fig
            draw_log(args, epoch, epoch_list, lossTr_list, Miou_list, lossVal_list)

            # save the model
            model_file_name = args.savedir + '/best_model.pth'
            last_model_file_name = args.savedir + '/last_model.pth'
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            if Miou > Best_Miou:
                Best_Miou = Miou
                torch.save(state, model_file_name)
                recorder.record_best_epoch(epoch, Best_Miou, Pa)

            # early_stopping monitor
            early_stopping.monitor(monitor=Miou)
            if early_stopping.early_stop:
                print("Early stopping and Save checkpoint")
                if not os.path.exists(last_model_file_name):
                    torch.save(state, last_model_file_name)
                    torch.cuda.empty_cache()  # empty_cache

                    loss, FWIoU, Miou, Miou_Noback, PerCiou_set, Pa, PerCpa_set, Mpa, MF, F_set, F1_Noback = \
                        predict_multiscale_sliding(args=args, model=model,
                                                   testLoader=testLoader,
                                                   scales=[1.0],
                                                   overlap=0.3,
                                                   criterion=criterion,
                                                   mode=args.predict_type,
                                                   save_result=False)
                    print("Epoch {}  lr= {:.6f}  Train Loss={:.4f}  Val Loss={:.4f}  Miou={:.4f}  PerCiou_set={}\n"
                          .format(epoch, lr, lossTr, loss, Miou, str(PerCiou_set)))
                break

            total_second = start_time + (args.max_epochs - epoch) * train_per_epoch_seconds + \
                           ((args.max_epochs - epoch) / args.val_epochs + 10) * validation_per_epoch_seconds + 43200
            print('Best Validation MIoU:{}'.format(Best_Miou))
            print('Training deadline is: {}\n'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(total_second))))


def parse_args():
    parser = ArgumentParser(description='Semantic segmentation with pytorch')
    # model and dataset
    parser.add_argument('--model', type=str, default="Deeplabv3plus_res50", help="model name")
    parser.add_argument('--backbone', type=str, default="resnet50", help="backbone name")
    parser.add_argument('--dataset', type=str, default="EPFL", choices=['EPFL', 'comballaz'],
                        help="dataset to train: EPFL or comballaz")
    parser.add_argument('--pretrained', type=bool, default=True,
                        help="whether choice backbone pretrained on imagenet")
    parser.add_argument('--normalization', type=bool, default=True,
                        help="whether normalize the input image")
    parser.add_argument('--img_path', type=str, default='./datasets/EPFL/val_sim',
                        help="Directory where the raw images are in")
    parser.add_argument('--out_stride', type=int, default=16, help="output stride of backbone")
    parser.add_argument('--mult_grid', action='store_true',
                        help="whether choice mult_grid in backbone last layer")
    parser.add_argument('--root', type=str, default="", help="path of datasets")
    parser.add_argument('--random_scale', type=bool, default=True, help="input image resize 0.75 to 2")
    parser.add_argument('--num_workers', type=int, default=16, help=" the number of parallel threads")
    # training hyper params
    parser.add_argument('--max_epochs', type=int, default=50, help="the number of epochs: 300 for train")
    parser.add_argument('--batch_size', type=int, default=4, help="the batch size is set to 16 GPU")
    parser.add_argument('--val_epochs', type=int, default=5, help="the number of epochs: 100 for val set")
    parser.add_argument('--lr', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--optim', type=str.lower, default='adam', choices=['sgd', 'adam', 'adamw'],
                        help="select optimizer")
    parser.add_argument('--predict_type', default="validation", choices=["validation", "predict"],
                        help="Defalut use validation type")
    parser.add_argument('--tile_hw_size', type=str, default='480, 720',
                        help=" the tile_size is when evaluating or testing")
    parser.add_argument('--flip_merge', action='store_true', help="Defalut use predict without flip_merge")
    parser.add_argument('--loss', type=str, default="CrossEntropyLoss2d",
                        choices=['CrossEntropyLoss2d', 'DiceLoss', 'FocalLoss2d'],
                        help="choice loss for train or val in list")
    # cuda setting
    parser.add_argument('--cuda', type=bool, default=True, help="running on CPU or GPU")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpus_id', type=str, default="0", help="default GPU devices 0")
    # checkpoint and log
    parser.add_argument('--resume', type=str,
                        default='/media/shanci/Samsung_T5/checkpoint/Deeplabv3plus_res50/CrossEntropyLoss2d/best_model.pth',
                        help="use this file to load last checkpoint for continuing training")
    parser.add_argument('--savedir', default="/media/shanci/Samsung_T5/checkpoint/",
                        help="directory to save the model snapshot")
    parser.add_argument('--logFile', default="log.txt", help="storing the training and validation logs")
    parser.add_argument('--classes', default=6, help="number of classes")
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    main(args)

    end = time.time()
    hour = 1.0 * (end - start) / 3600
    minute = (hour - int(hour)) * 60
    if args.local_rank == 0:
        print("training time: %d hour %d minutes" % (int(hour), int(minute)))
