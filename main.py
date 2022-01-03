import argparse
import os
import time
from trainer import Trainer

def str2bool(v):
    return v.lower() in ('true')

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train'  , action='store_true', default=False, help='initial training')
    parser.add_argument('--restore', action='store_true', default=False, help='the latest model weights')

    parser.add_argument('--gan'        , type=bool,  default=False,  help='use generative adversarial nets')
    parser.add_argument('--version'    , type=str,   default='V01', help='marked by num')
    parser.add_argument('--epoch_start', type=int,   default=0,     help='The start of epochs to run')
    parser.add_argument('--epoch'      , type=int,   default=200,   help='The number of epochs to run')
    parser.add_argument('--valid_ratio', type=float, default=0.1,   help='The valid ratio')

    parser.add_argument('--inputs_path', type=str, default='../data_YE/Qz0505_Raw_shot600_Nr360_Lt1201.segy')
    parser.add_argument('--labels_path', type=str, default='../data_YE/Qz0505_SRME_shot600_Nr360_Lt1201.segy')
    parser.add_argument('--preds_file' , type=str, default='Qz0505_NETS_shot600_Nr360_Lt1201.segy')
    parser.add_argument('--mute'       , type=int, default=0, help='The mute sampling length')

    parser.add_argument('--batch_size'      , type=int, default=1,  help='The size of batch per gpu')
    parser.add_argument('--train_print_freq', type=int, default=10, help='The number of train_print_freqy')
    parser.add_argument('--valid_print_freq', type=int, default=10, help='The number of valid_print_freqy')
    parser.add_argument('--test_print_freq' , type=int, default=1,  help='The number of test_print_freqy')

    if parser.get_default('gan'):
        parser.add_argument('--mark'  , type=str,   default='_gan')
        parser.add_argument('--g_name', type=str,   default='ATTN_UNET',  help='ATTN_UNET / UNET / ATTN_CAE / CAE')
        parser.add_argument('--d_name', type=str,   default='ATTN_CNN',   help='ATTN_CNN / CNN')
        parser.add_argument('--g_opt' , type=str,   default='adam',       help='optimizer for generator')
        parser.add_argument('--d_opt' , type=str,   default='adam',       help='optimizer for discriminator')
        parser.add_argument('--g_lr'  , type=float, default=0.001,        help='learning rate for generator')
        parser.add_argument('--d_lr'  , type=float, default=0.001,        help='learning rate for discriminator')
        parser.add_argument('--beta1' , type=float, default=0.9,          help='beta1 for Adam optimizer')
        parser.add_argument('--beta2' , type=float, default=0.999,        help='beta2 for Adam optimizer')

        parser.add_argument('--ver_dir'   , type=str, default=os.path.join('..', parser.get_default('version') + parser.get_default('mark')),   help='Root directory name to save')
        parser.add_argument('--model_dir' , type=str, default=os.path.join(parser.get_default('ver_dir'), 'model'), help='Directory name to save the checkpoints')
        parser.add_argument('--g_ckpt_dir', type=str, default=os.path.join(parser.get_default('model_dir'), 'g_'+parser.get_default('g_name')), help='Directory name to save the checkpoints')
        parser.add_argument('--d_ckpt_dir', type=str, default=os.path.join(parser.get_default('model_dir'), 'd_'+parser.get_default('d_name')), help='Directory name to save the checkpoints')

    else:
        parser.add_argument('--mark' , type=str,   default='')
        parser.add_argument('--name' , type=str,   default='ATTN_UNET', help='ATTN_UNET / UNET / ATTN_CAE / CAE / ATTN_CNN / CNN')
        parser.add_argument('--opt'  , type=str,   default='adam',      help='optimizer')
        parser.add_argument('--lr'   , type=float, default=0.001,       help='learning rate')
        parser.add_argument('--beta1', type=float, default=0.9,         help='beta1 for Adam optimizer')
        parser.add_argument('--beta2', type=float, default=0.999,       help='beta2 for Adam optimizer')

        parser.add_argument('--ver_dir'  , type=str, default=os.path.join('..', parser.get_default('version') + parser.get_default('mark')), help='Root directory name to save')
        parser.add_argument('--model_dir', type=str, default=os.path.join(parser.get_default('ver_dir'), 'model'), help='Directory name to save the checkpoints')
        parser.add_argument('--ckpt_dir' , type=str, default=os.path.join(parser.get_default('model_dir'), parser.get_default('name')), help='Directory name to save the checkpoints')

    parser.add_argument('--log_dir'  , type=str, default=os.path.join(parser.get_default('ver_dir'), 'log'),   help='Directory name to save training logs')
    parser.add_argument('--test_dir' , type=str, default=os.path.join(parser.get_default('ver_dir'), 'test'),  help='Directory name to save the generated images')
    parser.add_argument('--valid_dir', type=str, default=os.path.join(parser.get_default('ver_dir'), 'valid'), help='Directory name to save the valids on training')
    parser.add_argument('--img_dir'  , type=str, default=os.path.join(parser.get_default('ver_dir'), 'img'),   help='Directory name to save the visualized information on training')

    return parser.parse_args()

def main():
    #print args
    args = parse_args()
    if os.path.exists('../args'):
        pass
    else:
       os.makedirs('../args')

    #create dirs
    for _dir in [args.ver_dir, args.model_dir, args.log_dir, args.test_dir, args.img_dir]:
        if os.path.exists(_dir):
            pass
        else:
            os.makedirs(_dir)
    if args.gan:
        for _dir in [args.g_ckpt_dir, args.d_ckpt_dir]:
            if os.path.exists(_dir):
                pass
            else:
                os.makedirs(_dir)
    else:
        if os.path.exists(args.ckpt_dir):
            pass
        else:
            os.makedirs(args.ckpt_dir)

    #main
    trainer = Trainer(args)
    if args.train:
        args_path = os.path.join('../args', 'args_'+args.version+'_'+time.strftime('%Y%m%d_%H%M%S'))
        args_file = open(args_path, 'w')
        for k in list(vars(args).keys()):
            print('{:20s}     {:}'.format(k, vars(args)[k]), file=args_file)
        args_file.close()
        if args.gan:
            trainer.train_gan()
        else:
            trainer.train()
    else:
        trainer.test()

if __name__ == '__main__':
    main()
