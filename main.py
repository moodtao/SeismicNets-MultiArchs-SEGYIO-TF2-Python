import argparse
import os
import time
from trainer import Trainer

def str2bool(v):
    return v.lower() in ('true')

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--train'  , action='store_true', default=False)  # 训练标识，可外部控制，尽量不在参数卡中更改
    parser.add_argument('--restore', action='store_true', default=False)  # 加载已训练的模型，可外部控制，尽量不在参数卡中更改

    parser.add_argument('--version',     type=str,   default='V10')  # 程序版本号，做多次训练的标识，可任意更改
    parser.add_argument('--epoch'      , type=int,   default=300)    # 网络学习次数，可根据数据大小更改
    parser.add_argument('--valid_ratio', type=float, default=0.8)    # 网络验证集占数据集的比例（验证集不参与网络训练）
    parser.add_argument('--mute'       , type=int,   default=0)      # 时间采样点截断长度，在此点数之前的信号不参与训练

    parser.add_argument('--inputs_path', type=str, default='../data/ShangHe_Multi_Wave_before.segy')  # 输入数据路径
    parser.add_argument('--labels_path', type=str, default='../data/ShangHe_Multi_Wave_after.segy')   # 标签数据路径
    parser.add_argument('--tests_path',  type=str, default='../data/ShangHe_Multi_Wave_before.segy')  # 测试输入数据路径
    parser.add_argument('--preds_file',  type=str, default='ShangHe_Multi_Wave_UNET.segy')  # 测试输出数据的【名称】，不是路径！会自动保存在test文件夹中！


    parser.add_argument('--model_save_freq',  type=int, default=100) # 网络模型保存频率，默认为训练100个Epoch保存1次模型，可自由更改
    parser.add_argument('--train_print_freq', type=int, default=10)  # 网络学习信息输出频率，默认为训练10个道集输出1次损失函数值，可自由更改
    parser.add_argument('--valid_print_freq', type=int, default=10)  # 网络验证信息输出频率，默认为验证10个道集输出1次信噪比值，可自由更改
    parser.add_argument('--test_print_freq' , type=int, default=1)   # 网络测试信息输出频率，默认为测试1个道集输出1次信噪比值，可自由更改


#=======以下内容尽量不作更改========#
    parser.add_argument('--name',             type=str,   default='ATTN_UNET') # 网络名称标识，不作更改
    parser.add_argument('--opt',              type=str,   default='adam')      # 优化器名称，不作更改
    parser.add_argument('--lr',               type=float, default=0.0001)      # 网络学习率，不作更改
    parser.add_argument('--beta1',            type=float, default=0.9)         # 优化器超参数，不作更改
    parser.add_argument('--beta2',            type=float, default=0.999)       # 优化器超参数，不作更改

    parser.add_argument('--ver_dir',   type=str, default=os.path.join('..', parser.get_default('version')), help='Root directory name to save')
    parser.add_argument('--model_dir', type=str, default=os.path.join(parser.get_default('ver_dir'), 'model'), help='Directory name to save the checkpoints')
    parser.add_argument('--ckpt_dir',  type=str, default=os.path.join(parser.get_default('model_dir'), parser.get_default('name')), help='Directory name to save the checkpoints')

    parser.add_argument('--log_dir',   type=str, default=os.path.join(parser.get_default('ver_dir'), 'log'),   help='Directory name to save training logs')
    parser.add_argument('--test_dir',  type=str, default=os.path.join(parser.get_default('ver_dir'), 'test'),  help='Directory name to save the generated images')
    parser.add_argument('--img_dir',   type=str, default=os.path.join(parser.get_default('ver_dir'), 'img'),   help='Directory name to save the visualized information on training')

    return parser.parse_args()

def main():
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
        trainer.train()
    else:
        trainer.test()

if __name__ == '__main__':
    main()
