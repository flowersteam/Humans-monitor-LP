import matplotlib.pyplot as plt
import numpy as np
from os import path

from python_scripts.utils import loc_utils as lut
from python_scripts.utils import vis_utils as vut
from python_scripts.utils.vis_utils import gcolors, fullglabels, colors, gmarkers


def make_fig(data_path, figname, save_to, save_as=None):
    cols = ['st1', 'st2', 'st3', 'st4']
    df = lut.unpickle(data_path)
    df = df.set_index(['grp', 'sid', 'trial']).loc[:, cols]

    counts_per_sid = df.groupby(['grp', 'sid']).sum()
    export_data = (counts_per_sid / 250).reset_index()
    print(export_data.head())
    # export_data.to_csv('data/selection_proportions.csv', index=False)

    mean_counts_per_grp = counts_per_sid.groupby('grp').agg(['mean', 'sem'])
    print(mean_counts_per_grp)

    fig = plt.figure(figname, figsize=[5, 4])
    ax = vut.pretty(fig.add_subplot(111), 'y')
    ax.axhline(25, ls='--', color='k', alpha=.8)
    x = np.array([1, 2, 3, 4])
    for i, grp in enumerate([1, 0]):
        y = mean_counts_per_grp.loc[grp, (slice(None), 'mean')] / 250 * 100
        yerr = mean_counts_per_grp.loc[grp, (slice(None), 'sem')] / 250 * 100
        ax.errorbar(x, y, yerr=yerr, color=gcolors[grp], marker=gmarkers[grp],
                    capsize=5, markersize=8, lw=2, label=fullglabels[grp])

    ax.set_xticks(x)
    ax.set_xticklabels(['A1', 'A2', 'A3', 'A4'], fontsize=14, fontweight='bold')
    for xt, c in zip(ax.get_xticklabels(), colors):
        xt.set_color(c)

    ax.set_xlabel('Learning activity', fontsize=14)
    ax.set_ylabel('Trials per activity\n(%; Mean and SEM)', fontsize=14)
    leg = ax.legend(fontsize=14)

    fig.tight_layout()
    if save_as:
        vut.save_it(fig, save_to, figname, save_as=save_as, compress=False)


if __name__ == '__main__':
    data_path = path.join(path.dirname(__file__), '../data/ntm_data_freeplay.pkl')
    save_path = path.join(path.dirname(__file__), '../figures')

    make_fig(
        data_path=data_path,
        figname='figure2a',
        save_to=save_path,
        save_as='svg'
    )