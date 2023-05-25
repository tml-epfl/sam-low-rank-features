import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_loss(losses, training, random_step_fgsm, epoch, loss_type, i_ex):
    sns.set()
    plot_name = 'training={}_random_step_fgsm={}_loss_epoch={}_i_ex={}_type={}'.format(
        training, random_step_fgsm, epoch, i_ex, loss_type)
    marker_size, line_width = 5.0, 0.5
    interp_vals = np.arange(len(losses))
    x_values = (interp_vals - max(interp_vals)/2) / (max(interp_vals)/2)
    ax = sns.lineplot(x_values, losses, linewidth=line_width,
                      marker='o', markersize=marker_size, color="black")
    ax.set_xlabel('Interpolation coefficient')
    ax.set_ylabel('Adversarial loss')
    # ax.legend(loc='best', prop={'size': 12})
    ax.set_title(plot_name)
    plt.savefig('plots/{}.pdf'.format(plot_name), bbox_inches='tight')
    plt.close()


def histogram_delta(delta, attack, rs_train, rs_attack):
    sns.set()
    plot_name = 'histogram_delta-attack={}-rs_train={}-rs_attack={}'.format(attack, rs_train, rs_attack)
    sns.distplot(delta.flatten().cpu(), kde=False, rug=False, hist_kws={'log': True})
    plt.savefig('plots/{}.pdf'.format(plot_name), bbox_inches='tight')
    plt.close()

