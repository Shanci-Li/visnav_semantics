import os
import torch
from baseline.builders.model_builder import build_model


models = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


# def get_model(name, num_cls=19, **args):
#     net = models[name](num_cls=num_cls, **args)
#     if torch.cuda.is_available():
#         net = net.cuda()
#     return net


def get_model(args):
    if args.fs_model:
        if os.path.isfile(args.fs_model):
            net = build_model('Deeplabv3plus_res50', 6, output_feature=args.discrim_feat)
            checkpoint = torch.load(args.fs_model)['model']
            check_list = [i for i in checkpoint.items()]
            # Read weights with multiple cards, and continue training with a single card this time
            if 'module.' in check_list[0][0]:
                new_stat_dict = {}
                for k, v in checkpoint.items():
                    new_stat_dict[k[7:]] = v
                net.load_state_dict(new_stat_dict, strict=True)
            # Read the training weight of a single card, and continue training with a single card this time
            else:
                net.load_state_dict(checkpoint)

            net.cuda()
        else:
            print("no fs net found at '{}'".format(args.fs_model))
            raise FileNotFoundError("no fs net found at '{}'".format(args.fs_model))

    else:
        raise FileNotFoundError("no fs net found")
    return net