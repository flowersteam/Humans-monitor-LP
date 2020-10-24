import matplotlib.pyplot as plt
from matplotlib import gridspec, lines, legend_handler
import numpy as np
from os import path
from statsmodels.formula.api import ols
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.stats as scs


from python_scripts.utils import loc_utils as lut
from python_scripts.utils import vis_utils as vut
from python_scripts.utils.vis_utils import ncolors, fullglabels, glabels


def make_fig(data_path, figname, save_to, save_as=''):
    xstr = 'sc_lep'
    df = lut.unpickle(data_path).sort_values(by=xstr)
    df = df.loc[df.ntm > 0, :]

    propS = np.sum(df.grp == 1) / df.shape[0]

    fig = plt.figure(figname, figsize=[7, 7])
    gs = gridspec.GridSpec(2, 2)

    # Figure (scatter plot and histograms)
    # ====================================
    ghost_top = fig.add_subplot(gs[0, 0])
    ghost_top.set_ylabel('Relative frequency', fontsize=14, labelpad=30)
    ghost_top.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ghost_top.spines.values(): spine.set_visible(False)

    ax_top1 = vut.pretty(inset_axes(ghost_top, width='100%', height='30%', loc=9, borderpad=0))
    ax_top2 = vut.pretty(inset_axes(ghost_top, width='100%', height='30%', loc=10, borderpad=0))
    ax_top3 = vut.pretty(inset_axes(ghost_top, width='100%', height='30%', loc=8, borderpad=0))

    ghost_right = fig.add_subplot(gs[1, 1])
    ghost_right.set_xlabel('Relative frequency', fontsize=14, labelpad=30)
    ghost_right.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ghost_right.spines.values(): spine.set_visible(False)
    ax_right1 = vut.pretty(inset_axes(ghost_right, width='30%', height='100%', loc=6, borderpad=0))
    ax_right2 = vut.pretty(inset_axes(ghost_right, width='30%', height='100%', loc=10, borderpad=0))
    ax_right3 = vut.pretty(inset_axes(ghost_right, width='30%', height='100%', loc=7, borderpad=0))

    ax_scat = vut.pretty(fig.add_subplot(gs[1, 0]))

    bins = np.arange(0, 1.02, .1)
    labels = {'ntm': ['NAM ' + str(i) for i in (0, 1, 2, 3)], 'grp': fullglabels}
    axes_by_ntm = {
        'top': {
            3: ax_top1,
            2: ax_top2,
            1: ax_top3},
        'right': {
            1: ax_right1,
            2: ax_right2,
            3: ax_right3}}
    for ntm in [1, 2, 3]:
        axes_by_ntm['top'][ntm].set_xlim(.0, .8)
        axes_by_ntm['top'][ntm].set_ylim(0., .4)
        axes_by_ntm['right'][ntm].set_xlim(0., .45)
        axes_by_ntm['right'][ntm].set_ylim(0.38, 1.)
        for grp in [0, 1]:
            x = df.loc[(df.ntm == ntm) & (df.grp == grp), xstr]
            y = df.loc[(df.ntm == ntm) & (df.grp == grp), 'post']
            ax_scat.scatter(x, y, s=30, alpha=.7,
                            facecolors=ncolors[ntm - 1] if grp else 'w',
                            edgecolors=ncolors[ntm - 1])

            rf, _ = np.histogram(x, bins=bins, weights=np.ones_like(x) / x.size)
            axes_by_ntm['top'][ntm].plot(bins[:-1], rf, c=ncolors[ntm - 1], lw=2,
                                         ls='-' if grp else '--',
                                         label='{} / NAM-{}'.format(glabels[grp], ntm))
            axes_by_ntm['top'][ntm].tick_params(labelbottom=False)
            axes_by_ntm['top'][ntm].text(.02, .9, 'NAM-{}'.format(ntm), ha='left', va='top', fontsize=12,
                                         color=ncolors[ntm - 1], transform=axes_by_ntm['top'][ntm].transAxes)

            rf, _ = np.histogram(y, bins=bins, weights=np.ones_like(x) / x.size)
            axes_by_ntm['right'][ntm].plot(rf, bins[1:], c=ncolors[ntm - 1], lw=2,
                                           ls='-' if grp else '--',
                                           label='{} / NAM-{}'.format(glabels[grp], ntm))
            axes_by_ntm['right'][ntm].tick_params(labelleft=False)
            axes_by_ntm['right'][ntm].text(.95, .02, 'NAM-{}'.format(ntm), ha='right', va='bottom', fontsize=12,
                                           color=ncolors[ntm - 1], transform=axes_by_ntm['right'][ntm].transAxes)

    ax_scat.set_xlim(.0, .8)
    ax_scat.set_ylim(0.38, 1.)
    ax_scat.set_xlabel('Self-challenge (SC)', fontsize=14)
    ax_scat.set_ylabel('Weighted performance (dwfPC)', fontsize=14)

    # Edit legend
    c = 'darkgray'
    mark_ig = lines.Line2D([0], [0], ls='', marker='o', label=fullglabels[0], markerfacecolor='w', markeredgecolor=c)
    line_ig = lines.Line2D([0], [0], color=c, lw=2, label=fullglabels[0], ls='--', dashes=(2, 1))

    mark_eg = lines.Line2D([0], [0], color=c, ls='', marker='o', label=fullglabels[1])
    line_eg = lines.Line2D([0], [0], color=c, lw=2, label=fullglabels[1])

    ax_scat.legend(((line_ig, mark_ig), (line_eg, mark_eg)), fullglabels.values(),
                   bbox_to_anchor=(.5, 1.1,),
                   fontsize=12, ncol=3, loc='center',
                   handler_map={tuple: legend_handler.HandlerTuple(ndivide=None)})

    # Main Model
    # ====================================
    print(df.columns)
    qreg = ols('post ~ (pre + {0} + np.power({0}, 2) + grp)'.format(xstr), data=df).fit()
    #     print(qreg.summary())
    x = np.linspace(df.loc[:, xstr].min(), df.loc[:, xstr].max(), 100)
    y_hat = qreg.get_prediction({xstr: x,
                                 'pre': np.full_like(x, df.pre.mean()),
                                 'grp': np.full_like(x, propS)
                                 }).summary_frame()
    print(y_hat.head())
    c, alpha = 'k', .7
    ax_scat.plot(x, y_hat['mean'], c=c, alpha=alpha)
    ax_scat.plot(x, y_hat['mean_ci_lower'], c=c, lw=1, ls='--', alpha=alpha)
    ax_scat.plot(x, y_hat['mean_ci_upper'], c=c, lw=1, ls='--', alpha=alpha)

    # Standardize x before fitting the quadratic model
    df.loc[:, xstr] = scs.stats.zscore(df.loc[:, xstr])
    qreg = ols('post ~ (pre + {0} + np.power({0}, 2) + grp)'.format(xstr), data=df).fit()
    print(qreg.summary())
    nonqreg = ols('post ~ (pre + {0} + grp)'.format(xstr), data=df).fit()
    print(qreg.aic - nonqreg.aic)
    # ====================================

    # Supplementary Model
    # ====================================
    lreg = ols('{} ~ C(grp) * C(ntm)'.format(xstr), data=df).fit()
    print(lreg.summary())
    # ====================================

    print(df.groupby(['grp', 'ntm']).agg('count'))
    # Save figure
    fig.tight_layout()
    if save_as:
        vut.save_it(fig, save_to, figname=figname, save_as=save_as, compress=False, dpi=100)


if __name__ == '__main__':
    data_path = path.join(path.dirname(__file__), '../data/lpreds_data.pkl')
    save_path = path.join(path.dirname(__file__), '../figures')

    make_fig(
        data_path=data_path,
        figname='figure3',
        save_to=save_path,
        save_as='svg')
