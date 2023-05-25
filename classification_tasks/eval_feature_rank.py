import argparse
import os
import numpy as np
import torch
import time
import data
import models
import utils_eval
import losses


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar100', 'cifar10_horse_car', 'cifar10_dog_cat', 'uniform_noise'], type=str)
    parser.add_argument('--model', default='resnet18_plain', choices=['resnet18_plain', 'resnet18'], type=str)
    parser.add_argument('--set', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--model_path', default='', type=str, help='model name')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--n_eval', default=-1, type=int, help='#examples to evaluate on')
    parser.add_argument('--batch_size_eval', default=1024, type=int, help='batch size')
    parser.add_argument('--model_width', default=64, type=int, help='model width (# conv filters on the first layer for ResNets)')
    return parser.parse_args()


start_time = time.time()
args = get_args()
rho = args.model_path.split('sam_rho=')[1].split(' ')[0]
n_cls = 100 if args.dataset == 'cifar100' else 10
scaler = torch.cuda.amp.GradScaler(enabled=False)
loss_f = losses.cross_entropy()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

np.set_printoptions(precision=4, suppress=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

model = models.get_model(args.model, n_cls, False, data.shapes_dict[args.dataset], args.model_width).cuda().eval()

model_dict = torch.load('models/{}.pth'.format(args.model_path))['last']
model.load_state_dict({k: v for k, v in model_dict.items()})

# important to exclude the validation samples to get the correct training error
n_val = int(0.001 * data.shapes_dict[args.dataset][0])
val_indices = np.random.permutation(data.shapes_dict[args.dataset][0])[:n_val]
train_batches = data.get_loaders(args.dataset, args.n_eval, args.batch_size_eval, split='train', shuffle=False,
                                 data_augm=False, drop_last=False, p_label_noise=0.0, val_indices=val_indices)
test_batches = data.get_loaders(args.dataset, args.n_eval, args.batch_size_eval, split='test', shuffle=False,
                                data_augm=False, drop_last=False, p_label_noise=0.0)


train_err, train_loss, _ = utils_eval.rob_err(train_batches, model, 0, 0, scaler, 0, 0)
test_err, test_loss, _ = utils_eval.rob_err(test_batches, model, 0, 0, scaler, 0, 0)
print('test_err={:.2%}, train_err={:.2%}, train_loss={:.5f}'.format(test_err, train_err, train_loss))


feature_sing_vals, avg_sparsities, ns_active_relus_0p, ns_active_relus_1p, ns_active_relus_5p, ns_active_relus_10p = [], [], [], [], [], []

for i in [1, 2, 3, 4, 5]:
    feature_sing_vals += [utils_eval.compute_feature_sing_vals(train_batches, model, return_block=i)]  

    phi = utils_eval.compute_feature_matrix(train_batches, model, return_block=i)
    relu_threshold = phi.max() / 20
    avg_sparsities += [(phi > relu_threshold).mean()]
    ns_active_relus_0p += [((phi > relu_threshold).sum(0) > phi.shape[0] * 0.0).sum()]
    ns_active_relus_1p += [((phi > relu_threshold).sum(0) > phi.shape[0] * 0.01).sum()]
    ns_active_relus_5p += [((phi > relu_threshold).sum(0) > phi.shape[0] * 0.05).sum()]
    ns_active_relus_10p += [((phi > relu_threshold).sum(0) > phi.shape[0] * 0.1).sum()]

metrics = {
    'rho': rho,
    'test_err': test_err,
    'feature_ranks': [np.sum(np.cumsum(svals**2) <= np.sum(svals**2) * 0.99) + 1 for svals in feature_sing_vals],
    'avg_sparsities': avg_sparsities,
    'ns_active_relus_0p': ns_active_relus_0p,
    'ns_active_relus_1p': ns_active_relus_1p,
    'ns_active_relus_5p': ns_active_relus_5p,
    'ns_active_relus_10p': ns_active_relus_10p,
}
print(str(metrics) + ',')
time_elapsed = time.time() - start_time
print('Done in {:.2f}m'.format((time.time() - start_time) / 60))

