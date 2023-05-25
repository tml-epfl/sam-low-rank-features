import numpy as np
import torch
from fc_nets import FCNet2Layers, FCNet, compute_grad_matrix


def get_iters_eval(n_iter_power, x_log_scale, n_iters_first=100, n_iters_next=151):
    num_iter = int(10**n_iter_power) + 1

    iters_loss_first = np.array(range(n_iters_first))
    if x_log_scale:
        iters_loss_next = np.unique(np.round(np.logspace(0, n_iter_power, n_iters_next)))
    else:
        iters_loss_next = np.unique(np.round(np.linspace(0, num_iter, n_iters_next)))[:-1]
    iters_loss = np.unique(np.concatenate((iters_loss_first, iters_loss_next)))
    
    return num_iter, iters_loss


def get_data_two_layer_relu_net(n, d, m_teacher, init_scales_teacher, seed, clsf=False, clsf_mu=0.0, clsf_margin=0.0001, biases=False, act='relu'):
    np.random.seed(seed) 
    torch.manual_seed(seed) 

    n_test = 1000
    H = np.eye(d)
    X = torch.tensor(np.random.multivariate_normal(np.zeros(d), H, n)).float()
    if not clsf:
        X = X / torch.sum(X**2, 1, keepdim=True)**0.5
    X_test = torch.tensor(np.random.multivariate_normal(np.zeros(d), H, n_test)).float()
    if not clsf:
        X_test = X_test / torch.sum(X_test**2, 1, keepdim=True)**0.5

    # generate ground truth labels
    with torch.no_grad():
        net_teacher = FCNet2Layers(n_feature=d, n_hidden=m_teacher, biases=[True, True] if biases else [False, False], act=act)
        net_teacher.init_gaussian(init_scales_teacher)
        net_teacher.layer1.weight.data = net_teacher.layer1.weight.data / torch.sum((net_teacher.layer1.weight.data)**2, 1, keepdim=True)**0.5
        net_teacher.layer2.weight.data = torch.sign(net_teacher.layer2.weight.data)
        if clsf:
            X[:n//2] -= clsf_mu
            X[n//2:] += clsf_mu
            X_test[:n//2] -= clsf_mu
            X_test[n//2:] += clsf_mu

        y, y_test = net_teacher(X), net_teacher(X_test)

        if clsf:  # convert to -1 / 1
            # idx_train, idx_test = torch.abs(y.flatten()) > clsf_margin, torch.abs(y_test.flatten()) > clsf_margin
            # X, y, X_test, y_test = X[idx_train], y[idx_train], X_test[idx_test], y_test[idx_test]

            y[y < -clsf_margin], y[torch.abs(y) <= clsf_margin], y[y > clsf_margin] = -1, ((torch.randn((y[torch.abs(y) <= clsf_margin]).shape) > 0).float() - 0.5) * 2, 1
            y_test[y_test < -clsf_margin], y_test[torch.abs(y_test) <= clsf_margin], y_test[y_test > clsf_margin] = -1, ((torch.randn((y_test[torch.abs(y_test) <= clsf_margin]).shape) > 0).float() - 0.5) * 2, 1

            # y, y_test = ((y > 0).float() - 0.5) * 2, ((y_test > 0).float() - 0.5) * 2
        print('y', y[:20, 0])
    
    return X, y, X_test, y_test, net_teacher


def get_data_multi_layer_relu_net(n, d, m_teacher, init_scales_teacher, seed):
    np.random.seed(seed + 1) 
    torch.manual_seed(seed + 1) 

    n_test = 1000
    H = np.eye(d)
    X = torch.tensor(np.random.multivariate_normal(np.zeros(d), H, n)).float()
    X = X / torch.sum(X**2, 1, keepdim=True)**0.5
    X_test = torch.tensor(np.random.multivariate_normal(np.zeros(d), H, n_test)).float()
    X_test = X_test / torch.sum(X_test**2, 1, keepdim=True)**0.5

    # generate ground truth labels
    with torch.no_grad():
        net_teacher = FCNet(n_feature=d, n_hidden=m_teacher)
        net_teacher.init_gaussian(init_scales_teacher)
        y, y_test = net_teacher(X), net_teacher(X_test)
        print('y:', y[:, 0])
    
    return X, y, X_test, y_test, net_teacher


def effective_rank(v):
    v = v[v != 0]
    v /= v.sum()
    return -(v * np.log(v)).sum()


def rm_too_correlated(net, X, V, corr_threshold=0.99):
    V = V.T
    idx_keep = np.where((V > 0.0).sum(0) > 0)[0]
    V_filtered = V[:, idx_keep]  # filter out zeros
    corr_matrix = np.corrcoef(V_filtered.T) 
    corr_matrix -= np.eye(corr_matrix.shape[0])

    idx_to_delete, i, j = [], 0, 0
    while i != corr_matrix.shape[0]:
        if (np.abs(corr_matrix[i]) > corr_threshold).sum() > 0:
            corr_matrix = np.delete(corr_matrix, (i), axis=0)
            corr_matrix = np.delete(corr_matrix, (i), axis=1)
            # print('delete', j)
            idx_to_delete.append(j)
        else:
            i += 1
        j += 1
    assert corr_matrix.shape[0] == corr_matrix.shape[1]
    idx_keep = np.delete(idx_keep, [idx_to_delete])
    
    return V[:, idx_keep].T

def compute_grad_matrix_dim(net, X, corr_threshold=0.99):
    grad_matrix = compute_grad_matrix(net, X)
    grad_matrix_sq_norms = np.sum(grad_matrix**2, 0)
    m = 100
    v_j = []
    for j in range(m):
        v_j.append(grad_matrix_sq_norms[[j, m+j, 2*m+j]])  # matrix: w1, w2, w3, w4
    V = np.vstack(v_j)

    V_reduced = rm_too_correlated(net, X, V, corr_threshold=corr_threshold)
    grad_matrix_dim = V_reduced.shape[0]
    return grad_matrix_dim

def compute_hessian(net, X, y):
    def loss_function(*all_params):
        w, bw, v, bv = all_params
        loss_f = lambda y_pred, y: torch.mean((y_pred - y)**2)
        y_pred = F.relu(X @ w.T + bw) @ v.T + bv
        loss = loss_f(y_pred, y)
        return loss

    h = torch.autograd.functional.hessian(loss_function, tuple(p for p in net.parameters()))
    # TODO: unfinished; the Hessian function returns a list of matrices, but we need to compose a single matrix out of them

