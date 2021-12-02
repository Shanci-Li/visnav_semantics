import os
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
from torch.utils import data
from argparse import ArgumentParser
from prettytable import PrettyTable
from builders.model_builder import build_model
from model_data_loader import SenmanticData
from builders.validation_builder import predict_multiscale_sliding
from baseline.utils.losses.loss import CrossEntropyLoss2d, FocalLoss2d, DiceLoss


def main(args):
    """
     main function for testing
     param args: global arguments
     return: None
    """
    t = PrettyTable(['args_name', 'args_value'])
    for k in list(vars(args).keys()):
        t.add_row([k, vars(args)[k]])
    print(t.get_string(title="Predict Arguments"))

    class_name = ['Sky', 'Ground', 'Vegetation', 'Buildings', 'Water', 'Bridges']
    label_index = [0, 1, 2, 3, 4, 5]
    class_dict_df = pd.DataFrame([class_name, label_index], index=['class_name', 'label_index']).T
    class_dict_df = class_dict_df.set_index('label_index')

    # build the model
    model = build_model(args.model, args.classes, args.backbone, args.pretrained, args.out_stride, args.mult_grid)

    # load the test set
    # test_path = './datasets/EPFL/test_oop_drone_real'
    test_path = args.test_dir
    args.img_path = test_path
    test_set = SenmanticData(test_path, augmentation=False)

    DataLoader = data.DataLoader(test_set, batch_size=args.batch_size,
                shuffle=False, num_workers=args.batch_size, pin_memory=True, drop_last=False)

    if args.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        model = model.cuda()
        cudnn.benchmark = True
        if not torch.cuda.is_available():
            raise Exception("no GPU found or wrong gpu id, please run without --cuda")

    args.save_seg_dir = os.path.join(args.save_seg_dir, test_path.split('/')[-1])

    if not os.path.exists(args.save_seg_dir):
        os.makedirs(args.save_seg_dir)

    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            checkpoint = torch.load(args.checkpoint)['model']
            check_list = [i for i in checkpoint.items()]
            # Read weights with multiple cards, and continue training with a single card this time
            if 'module.' in check_list[0][0]:
                new_stat_dict = {}
                for k, v in checkpoint.items():
                    new_stat_dict[k[7:]] = v
                model.load_state_dict(new_stat_dict, strict=True)
            # Read the training weight of a single card, and continue training with a single card this time
            else:
                model.load_state_dict(checkpoint)
        else:
            print("no checkpoint found at '{}'".format(args.checkpoint))
            raise FileNotFoundError("no checkpoint found at '{}'".format(args.checkpoint))

    # define loss function, respectively
    # Default uses cross quotient loss function
    if args.loss == 'CrossEntropyLoss2d':
        criterion = CrossEntropyLoss2d()
    elif args.loss == 'DiceLoss':
        criterion = DiceLoss()
    elif args.loss == 'FocalLoss2d':
        criterion = FocalLoss2d()

    criterion = CrossEntropyLoss2d()
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n"
          ">>>>>>>>>>>  beginning testing   >>>>>>>>>>>>\n"
          ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    predict_multiscale_sliding(args=args, model=model, testLoader=DataLoader, class_dict_df=class_dict_df,
                                scales=args.scales, overlap=args.overlap, criterion=criterion,
                                mode=args.predict_type, save_result=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', default="Deeplabv3plus_res50", help="model name")
    parser.add_argument('--backbone', type=str, default="resnet50", help="backbone name")
    parser.add_argument('--pretrained', action='store_true',
                        help="whether choice backbone pretrained on imagenet")
    parser.add_argument('--test_dir', type=str, default='./datasets/EPFL/test_oop_drone_real',
                        help="Directory where the test data is")
    parser.add_argument('--augmentation', type=bool, default=False,
                        help="whether augment the input image")
    parser.add_argument('--out_stride', type=int, default=16, help="output stride of backbone")
    parser.add_argument('--mult_grid', action='store_true',
                        help="whether choice mult_grid in backbone last layer")
    parser.add_argument('--root', type=str, default="", help="path of datasets")
    parser.add_argument('--predict_type', default="validation", choices=["validation", "predict"],
                        help="Defalut use validation type")
    parser.add_argument('--flip_merge', action='store_true', help="Defalut use predict without flip_merge")
    parser.add_argument('--scales', type=float, nargs='+', default=[1.0], help="predict with multi_scales")
    parser.add_argument('--overlap', type=float, default=0.0, help="sliding predict overlap rate")
    parser.add_argument('--num_workers', type=int, default=4, help="the number of parallel threads")
    parser.add_argument('--batch_size', type=int, default=1,
                        help=" the batch_size is set to 1 when evaluating or testing NOTES:image size should fixed!")
    parser.add_argument('--tile_hw_size', type=str, default='480, 720',
                        help=" the tile_size is when evaluating or testing")
    parser.add_argument('--input_size', type=str, default=(480, 720),
                        help=" the input_size is for build ProbOhemCrossEntropy2d loss")
    parser.add_argument('--checkpoint', type=str,
                        default='/media/shanci/Samsung_T5/checkpoint/EPFL/Deeplabv3plus_res50'
                                '/CrossEntropyLoss2d/augmentation/with_real_data_inplace/best_model.pth',
                        help="use the file to load the checkpoint for evaluating or testing ")
    parser.add_argument('--img_path', type=str, default=None,
                        help="Directory where the raw images are in")
    parser.add_argument('--save_seg_dir', type=str, default=None,
                        help="saving path of prediction result")
    parser.add_argument('--loss', type=str, default="CrossEntropyLoss2d",
                        choices=['CrossEntropyLoss2d', 'DiceLoss', 'FocalLoss2d'],
                        help="choice loss for train or val in list")
    parser.add_argument('--cuda', default=True, help="run on CPU or GPU")
    parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--classes', default=6, help="number of classes")
    args = parser.parse_args()

    save_dirname = 'predict_results_oop_mIoU'

    args.save_seg_dir = os.path.join('/'.join(args.checkpoint.split('/')[:-1]), save_dirname)

    main(args)
