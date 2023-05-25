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
    parser.add_argument('--dataset', default='cifar10', choices=['mnist', 'svhn', 'cifar10', 'cifar10_horse_car', 'cifar10_dog_cat', 'uniform_noise'], type=str)
    parser.add_argument('--model', default='resnet18', choices=['resnet18', 'cnn', 'fc', 'linear', 'lenet'], type=str)
    parser.add_argument('--set', default='test', type=str, choices=['train', 'test'])
    parser.add_argument('--weight_modification', default='none', type=str, choices=['none', 'sam_m_full', 'sam_m_1'])
    parser.add_argument('--model_path', default='2020-05-01 19:16:54.509 dataset=cifar10 model=resnet18 eps=8.0 attack=fgsm attack_init=zero fgsm_alpha=1.25 epochs=30 pgd=2.0-10 grad_align_cos_lambda=0.2 cure_lambda=0.0 lr_max=0.3 seed=0 epoch=30',
                        type=str, help='model name')
    parser.add_argument('--model_short', default='noname', type=str, help='model name')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--n_eval', default=-1, type=int, help='#examples to evaluate on')
    parser.add_argument('--loss', default='ce', choices=['ce', 'ce_offset', 'gce', 'smallest_k_ce'], type=str, help='Loss type.')
    parser.add_argument('--sam_rho', default=0.2, type=float, help='step size for SAM (sharpness-aware minimization)')
    parser.add_argument('--p_label_noise', default=0.0, type=float, help='label noise for evaluation')
    parser.add_argument('--activation', default='relu', type=str, help='currently supported only for resnet. relu or softplus* where * corresponds to the softplus alpha')
    parser.add_argument('--pgd_rr_n_iter', default=50, type=int, help='pgd rr number of iterations')
    parser.add_argument('--pgd_rr_n_restarts', default=10, type=int, help='pgd rr number of restarts')
    parser.add_argument('--n_layers', default=1, type=int, help='#layers on each conv layer (for model in [fc, cnn])')
    parser.add_argument('--model_width', default=64, type=int, help='model width (# conv filters on the first layer for ResNets)')
    parser.add_argument('--n_filters_cnn', default=16, type=int, help='#filters on each conv layer (for model==cnn)')
    parser.add_argument('--n_hidden_fc', default=1024, type=int, help='#filters on each conv layer (for model==fc)')
    parser.add_argument('--batch_size_eval', default=512, type=int, help='batch size for evaluation')
    parser.add_argument('--half_prec', action='store_true', help='eval in half precision')
    parser.add_argument('--early_stopped_model', action='store_true', help='eval the best model according to pgd_acc evaluated every k iters (typically, k=200)')
    parser.add_argument('--eval_grad_norm', action='store_true', help='evaluate the gradient norm')
    parser.add_argument('--aa_eval', action='store_true', help='perform autoattack evaluation')
    return parser.parse_args()


start_time = time.time()
args = get_args()
eps = args.eps / 255
half_prec = args.half_prec  # for more reliable evaluations: keep in the single precision
print_stats = False
n_cls = 2 if args.dataset in ['cifar10_horse_car', 'cifar10_dog_cat'] else 10

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

np.set_printoptions(precision=4, suppress=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

scaler = torch.cuda.amp.GradScaler(enabled=False)
loss_dict = {
    'ce': losses.cross_entropy(),
    'ce_offset': losses.cross_entropy_with_offset(loss_offset=0.1),
    'gce': losses.generalized_cross_entropy(q=0.7),
    'smallest_k_ce': losses.smallest_k_cross_entropy(frac_rm_per_batch=0.0)
}
loss_f = loss_dict[args.loss]

model = models.get_model(args.model, n_cls, args.half_prec, data.shapes_dict[args.dataset], args.model_width, args.activation)
model = model.cuda().eval()

model_dict = torch.load('models/{}.pth'.format(args.model_path))
model_dict = model_dict['best'] if args.early_stopped_model else model_dict['last']
model.load_state_dict({k: v for k, v in model_dict.items() if 'model_preact_hl1' not in k})

opt = torch.optim.SGD(model.parameters(), lr=0, momentum=0.9)

# important to exclude the validation samples to get the correct training error
n_val = int(0.1 * data.shapes_dict[args.dataset][0])
val_indices = np.random.permutation(data.shapes_dict[args.dataset][0])[:n_val]
train_batches = data.get_loaders(args.dataset, args.n_eval, args.batch_size_eval, split='train', shuffle=False,
                                 data_augm=False, drop_last=False, p_label_noise=args.p_label_noise, val_indices=val_indices)
test_batches = data.get_loaders(args.dataset, args.n_eval, args.batch_size_eval, split='test', shuffle=False,
                                data_augm=False, drop_last=False, p_label_noise=args.p_label_noise)


# import ipdb;ipdb.set_trace()
if args.eval_grad_norm:
    n_ex, grad_norm_total = 0, 0.0
    for i, (X, X_augm2, y, _, ln) in enumerate(train_batches):
        X, y = X.cuda(), y.cuda()

        output = model(X)
        loss = loss_f(output, y)
        loss.backward()
        grad_norm = sum([torch.sum(p.grad**2) for p in model.parameters()])**0.5
        grad_norm_total += grad_norm.item()
        n_ex += y.size(0)
        opt.zero_grad()

    grad_norm_total /= n_ex
    print(grad_norm_total)


train_err, train_loss, _ = utils_eval.rob_err(train_batches, model, 0, 0, scaler, 0, 0)
test_err, test_loss, _ = utils_eval.rob_err(test_batches, model, 0, 0, scaler, 0, 0)

print('err={:.2%}/{:.2%}, loss={:.4f}/{:.4f}'.format(train_err, test_err, train_loss, test_loss))

if args.aa_eval:
    from autoattack import autoattack
    images, labels, _, _ = data.get_xy_from_loader(test_batches)
    adversary = autoattack.AutoAttack(model, norm='Linf', eps=eps)
    x_adv = adversary.run_standard_evaluation(images, labels, bs=args.batch_size_eval)


time_elapsed = time.time() - start_time
print('Done in {:.2f}m'.format((time.time() - start_time) / 60))

