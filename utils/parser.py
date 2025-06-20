import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', 
        type = str, 
        help = 'yaml config file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')     
    parser.add_argument('--local-rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=2)   
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')      
    # bn
    parser.add_argument(
        '--sync_bn', 
        action='store_true', 
        default=False, 
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--start_ckpts', type = str, default=None, help = 'reload used ckpt path')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--val_freq', type = int, default=1, help = 'test freq')
    parser.add_argument('--voting', action='store_true', default=False, help = 'turn on voting strategy for classification')
    parser.add_argument('--voting_thres', type = float, default=80.0, help = 'threshold for voting strategy')
    parser.add_argument(
        '--resume', 
        action='store_true', 
        default=False, 
        help = 'autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test', 
        action='store_true', 
        default=False, 
        help = 'test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model', 
        action='store_true', 
        default=False, 
        help = 'finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model', 
        action='store_true', 
        default=False, 
        help = 'training modelnet from scratch')
    parser.add_argument(
        '--eval_zeroshot', 
        action='store_true', 
        default=False, 
        help = 'run zero shot evaluation only, start_ckpts is needed')
    parser.add_argument(
        '--eval_type',
        choices=['2d3d', '2d', '3d', 'depth', 'depth2d3d', 'depth2d'],
        default='3d',
        help='zero shot evaluation type')   
    parser.add_argument(
        '--eval_linearsvm', 
        action='store_true', 
        default=False, 
        help = 'run linear svm evaluation only, start_ckpts is needed')
    parser.add_argument(
        '--mode', 
        choices=['easy', 'median', 'hard', None],
        default=None,
        help = 'difficulty mode for shapenet')        
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)
    parser.add_argument(
        '--amp', 
        action='store_true', 
        default=False, 
        help = 'for amp training')
    
    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while finetune_model mode')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' +args.mode
    args.experiment_path = os.path.join('./logs', Path(args.config).stem, Path(args.config).parent.stem, args.exp_name)
    args.tfboard_path = os.path.join('./logs', Path(args.config).stem, Path(args.config).parent.stem,'TFBoard' ,args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    if args.resume and args.start_ckpts is not None:
        ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
        if not os.path.exists(ckpt_path):
            args.resume = False
            print("Warning: resume and start_ckpts given together, checking resume checkpoint empty!")
            print("Starting from start_ckpts!!!!!")
        else:
            args.start_ckpts = None
            print("Warning: resume and start_ckpts given together, checking resume checkpoint not empty!")
            print("Starting from resume checkpoint!!!!!")
    if args.resume:
        ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
        if not os.path.exists(ckpt_path):
            args.resume = False
            print("Warning: check resume checkpoint empty, start training from scratch!")
    if args.eval_zeroshot and args.finetune_model:
        raise ValueError(
            "eval_zeroshot mode is conflict with finetune!")
    if args.eval_linearsvm and args.finetune_model:
        raise ValueError(
            "eval_linearsvm mode is conflict with finetune!")
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path, exist_ok=True)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path, exist_ok=True)
        os.makedirs(os.path.join(args.tfboard_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(args.tfboard_path, 'test'), exist_ok=True)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)

