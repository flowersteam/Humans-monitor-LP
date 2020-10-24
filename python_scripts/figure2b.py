import matplotlib.pyplot as plt
from matplotlib import gridspec, lines
import numpy as np
from os import path
from statsmodels.formula.api import ols

from python_scripts.utils import loc_utils as lut
from python_scripts.utils import vis_utils as vut
from python_scripts.utils.vis_utils import gcolors, fullglabels, tlabels, colors


def make_top_fig(data_path, figname, save_to, save_as=None):
    cols = ['st1', 'st2', 'st3', 'st4']
    df = lut.unpickle(data_path)
    _, Ns = np.unique(df.grp.values, return_counts=True)
    Ns = Ns // 250
    df = df.set_index(['grp', 'sid', 'trial']).loc[:, cols]
    counts_per_trial = df.groupby(['grp', 'trial']).sum()

    fig = plt.figure(figname, figsize=[5, 6])
    gs = gridspec.GridSpec(2, 1)

    # Ghost axis
    ax = fig.add_subplot(gs[:2, 0])
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylabel('Fraction of Ps selecting a learning activity', fontsize=14, labelpad=30)

    for i, grp in enumerate([1, 0]):
        # Left panels
        ax = vut.pretty(fig.add_subplot(gs[i, :3]))
        ax.set_xlim(1, 250)
        ax.set_ylim(.08, .5)
        ax.grid(True)

        if i:
            ax.set_xlabel('Trial', fontsize=14)
        else:
            ax.tick_params(labelbottom=False)
            handles = [lines.Line2D([0], [0], color=colors[k], ls='', marker='o', label=tlabels[k + 1]) for k in range(4)]
            leg = ax.legend(handles, tlabels.values(), handletextpad=.05,
                            bbox_to_anchor=(0, 1, 1, 0.2), loc='lower left', mode='expand', ncol=4, fontsize=14)
            vut.color_legend(leg)

        txt = ax.text(x=.05, y=.95, s=fullglabels[grp],
                      ha='left', va='top', transform=ax.transAxes,
                      color=gcolors[grp], fontweight='bold', fontsize=14)

        for tsk in [1, 2, 3, 4]:
            psel = counts_per_trial.loc[(grp, slice(None)), 'st' + str(tsk)].values.squeeze() / Ns[grp]
            ax.plot(psel, c=colors[tsk - 1], label=tlabels[tsk], lw=2)

    fig.tight_layout()
    if save_as:
        vut.save_it(fig, save_to, figname, save_as=save_as, compress=False)


def make_bottom_fig(data_path, figname, save_to, save_as=None):
    # Prepare data
    cols = 'sid,grp,ntm,trial,t0,cor'.split(',')
    df = lut.unpickle(data_path)[cols]
    df.loc[:, 'trial'] = df.trial + 1

    gdf = df.groupby(['grp', 'trial'])[['cor']].mean()
    gdf.loc[:, 'cor'] = gdf.cor * 100
    w = 15

    lmdf = gdf.reset_index()
    lm = ols('cor ~ trial*C(grp,Treatment(reference=1))', data=lmdf).fit()
    params = lm.params
    print(lm.summary())

    # Make figure
    fig = plt.figure(figname, figsize=[5, 3])
    ax = vut.pretty(fig.add_subplot(111))

    for grp in [0, 1]:
        x = np.arange(1, 251)
        y = gdf.loc[(grp, slice(None)), :].values.squeeze()
        ax.plot(x, y, color=gcolors[grp], ls='', alpha=.3, marker='.')

        x_ = np.array([1, 251])
        y_ = lm.predict({'grp': (grp, grp), 'trial': x_})
        ax.plot(x_, y_, color=gcolors[grp], lw=2, alpha=.9, label=fullglabels[grp])

        intercept = params[0] + params[1] * grp
        slope = params[2] + params[3] * grp
        pos = int(slope > 0)
        txt = 'Y = {:.3f} {} {:.3f}*X'.format(intercept, '-+'[pos], np.abs(slope))
        print('grp {}: {}'.format(grp, txt))

    ax.set_xlabel('Trial', fontsize=14)
    ax.set_ylabel('% correct', fontsize=14)

    leg = ax.legend(fontsize=14, ncol=2)
    vut.color_legend(leg)

    ax.set_ylim(55, 85)
    ax.set_xlim(1, 250)
    fig.tight_layout()

    if save_as:
        vut.save_it(fig, save_to, figname, save_as=save_as, compress=False)


if __name__ == '__main__':
    data_path = path.join(path.dirname(__file__), '../data/ntm_data_freeplay.pkl')
    save_path = path.join(path.dirname(__file__), '../figures')
    fig_format = 'svg'

    make_top_fig(
        data_path=data_path,
        figname='figure2b_top',
        save_to=save_path,
        save_as=fig_format
    )

    make_bottom_fig(
        data_path=data_path,
        figname='figure2b_bottom',
        save_to=save_path,
        save_as=fig_format
    )

