import argparse
from approach import VAE_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_name', help='name of the state dict to load')
    parser.add_argument('--data_root', default='MNIST', help='data set root')
    parser.add_argument('--data_type', default='MNIST', help='only available for MNIST')
    parser.add_argument('--img_size', type=int, default=28, help='size of the input data image')
    parser.add_argument('--h1_dim', required=True, type=int, help='hidden layer 1 dimension')
    parser.add_argument('--h2_dim', required=True, type=int, help='hidden layer 2 dimension')
    parser.add_argument('--z_dim', required=True, type=int, help='latent variable dimension')
    parser.add_argument('--cuda', type=bool, default=False, help='enable gpu computation')

    parser.add_argument('--niter', required=True, type=int, help='iterations to train')
    parser.add_argument('--batch_size', required=True, type=int, help='batch_size')
    parser.add_argument('--lr', required=True, type=float, help='learning rate')
    parser.add_argument('--save_name', help='name of the state dict to save')
    parser.add_argument('--log_dir', help='name of logdir for tensorboard')
    parser.add_argument('--start_epoch', type=int, default=0, help='the number of epoch to start')

    parser.add_argument('--sample_name', default='sample', help='name of the generated sample file')

    opt = parser.parse_args()

    model = VAE_model(opt.load_name, opt.data_root, opt.data_type, opt.img_size,
                      opt.h1_dim, opt.h2_dim, opt.z_dim, opt.cuda)
    model.train(opt.niter, opt.batch_size, opt.lr, opt.save_name, opt.log_dir, opt.start_epoch)

    model.test(opt.sample_name)


if __name__ == '__main__':
    main()
