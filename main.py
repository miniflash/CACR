import ast
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #data
    parser.add_argument('--phase', type=str, default='cacrtrain', choices=['pretrain','cacrtrain', 'cacreval'])
    parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
    parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting '
                                                              'Options:{0,1}')
    parser.add_argument('--data_path', type=str, default='./datasets/S3DIS/blocks_bs1_s1',
                                                    help='Directory to the source data')
    parser.add_argument('--pretrain_checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint of pre model for resuming')
    parser.add_argument('--model_checkpoint_path', type=str, default=None,
                        help='Path to the checkpoint of model for resuming')
    parser.add_argument('--save_path', type=str, default='./log_s3dis/',
                        help='Directory to the save log and checkpoints')
    parser.add_argument('--eval_interval', type=int, default=1500,
                        help='iteration/epoch inverval to evaluate model')

    #optimization
    parser.add_argument('--batch_size', type=int, default=32, help='Number of samples/tasks in one batch')
    parser.add_argument('--n_workers', type=int, default=16, help='number of workers to load data')
    parser.add_argument('--n_iters', type=int, default=30000, help='number of iterations/epochs to train')

    parser.add_argument('--lr', type=float, default=0.001,
                        help='Model (eg. protoNet or MPTI) learning rate [default: 0.001]')
    parser.add_argument('--step_size', type=int, default=5000, help='Iterations of learning rate decay')
    parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

    parser.add_argument('--pretrain_lr', type=float, default=0.001, help='pretrain learning rate [default: 0.001]')
    parser.add_argument('--pretrain_weight_decay', type=float, default=0., help='weight decay for regularization')
    parser.add_argument('--pretrain_step_size', type=int, default=50, help='Period of learning rate decay')
    parser.add_argument('--pretrain_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

    #few-shot episode setting
    parser.add_argument('--n_way', type=int, default=2, help='Number of classes for each episode: 1|3')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of samples/shots for each class: 1|5')
    parser.add_argument('--n_queries', type=int, default=1, help='Number of queries for each class')
    parser.add_argument('--n_episode_test', type=int, default=100,
                        help='Number of episode per configuration during testing')

    # Point cloud processing
    parser.add_argument('--pc_npts', type=int, default=2048, help='Number of input points for PointNet.')
    parser.add_argument('--pc_attribs', default='xyz',
                        help='Point attributes fed to PointNets, if empty then all possible. '
                             'xyz = coordinates, rgb = color, XYZ = normalized xyz')
    parser.add_argument('--pc_augm', action='store_true', help='Training augmentation for points in each superpoint')
    parser.add_argument('--pc_augm_scale', type=float, default=0,
                        help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
    parser.add_argument('--pc_augm_rot', type=int, default=1,
                        help='Training augmentation: Bool, random rotation around z-axis')
    parser.add_argument('--pc_augm_mirror_prob', type=float, default=0,
                        help='Training augmentation: Probability of mirroring about x or y axes')
    parser.add_argument('--pc_augm_jitter', type=int, default=1,
                        help='Training augmentation: Bool, Gaussian jittering of all attributes')

    # foreground adaptive alignment
    parser.add_argument('--max_per_class', type=int, default=256, help='Number of sampled points for alignment.')
    parser.add_argument('--sigma_multipliers', type=int, default=(0.5, 1.0, 2.0, 4.0), help='The init kernel bandwidth.')
    parser.add_argument('--num_neg', type=int, default=1, help='Number of negative classes.')
    parser.add_argument('--gamma_margin', type=int, default=0.1, help='Margin for negative and positive classes.')
    parser.add_argument('--lambda_ammd', type=int, default=1.0, help='Weight of fc-ammd loss.')

    # fine-grained features
    parser.add_argument('--output_dim', type=int, default=384,
                        help='The dimension of the final output of feature')
    parser.add_argument('--m_dim', type=int, default=128)
    parser.add_argument('--state_dim', type=int, default=128,)
    parser.add_argument('--head_dim', type=int, default=64)
    parser.add_argument('--chunk_size', type=int, default=64)

    # protoNet configuration
    parser.add_argument('--dist_method', default='cosine',
                        help='Method to compute distance between query feature maps and prototypes.[Option: cosine|euclidean]')

    # MPTI configuration
    parser.add_argument('--n_subprototypes', type=int, default=100,
                        help='Number of prototypes for each class in support set')
    parser.add_argument('--k_connect', type=int, default=200,
                        help='Number of nearest neighbors to construct local-constrained affinity matrix')
    parser.add_argument('--sigma', type=float, default=1., help='hyeprparameter in gaussian similarity function')

    args = parser.parse_args()
    args.pc_in_dim = len(args.pc_attribs)

    if args.phase=='cacrtrain':
        args.log_dir = args.save_path + 'log_cacrtrain_%s_S%d_N%d_K%d' %(args.dataset, args.cvfold,
                                                                             args.n_way, args.k_shot)
        from runs.cacr_train import train
        train(args)
    elif args.phase=='cacreval':
        args.log_dir = args.model_checkpoint_path
        from runs.eval import eval
        eval(args)
    elif args.phase=='pretrain':
        args.log_dir = args.save_path + 'log_pretrain_%s_S%d' % (args.dataset, args.cvfold)
        from runs.pre_train import pretrain
        pretrain(args)
    else:
        raise ValueError('Please set correct phase.')