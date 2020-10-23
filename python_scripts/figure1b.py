import matplotlib.pyplot as plt
import seaborn as sns
from os import path

from python_scripts.utils import loc_utils as lut
from python_scripts.utils import vis_utils as vut
from python_scripts.utils.vis_utils import gcolors, glabels, colors


def make_fig(data_path, figname, save_to='LOCAL', save_as=''):
    cols = ['sid', 'grp', 'tid', 'pc_first']
    df = lut.unpickle(data_path)
    df = df.loc[:, cols]
    df = df.replace({'1_1D': 1, '2_I1D': 2, '3_2D': 3, '4_R': 4, 'F': 0, 'S': 1})
    df['percent_correct'] = df.pc_first * 100

    fig = plt.figure(figname, figsize=[5, 5])

    ax = vut.pretty(fig.add_subplot(111))
    palet = sns.set_palette(sns.color_palette(gcolors))
    sns.boxplot(x='tid', y='percent_correct', hue='grp', data=df,
                saturation=1, palette=palet, ax=ax)

    props = {'connectionstyle': 'bar',
             'arrowstyle': '-',
             'patchA': None, 'patchB': None,
             'shrinkA': 10, 'shrinkB': 10,
             'linewidth': 2}
    bh, pad = 100, .025
    for i in range(3):
        ax.text(x=i + .5, y=bh + 15, s='***', fontsize=15, ha='center', va='top')
        ax.annotate('', xy=(i + pad, bh), xytext=(i + 1 - pad, bh), xycoords='data',
                    ha='center', va='top', arrowprops=props)

    ax.set_ylim(0, 115)
    ax.set_xlim(-.5, 3.8)
    ax.set_xlabel('Learning activity', fontsize=15)
    ax.set_ylabel('% correct', fontsize=20)
    ax.set_xticklabels(['A1', 'A2', 'A3', 'A4'], fontsize=15, fontweight='bold')
    for xt, c in zip(ax.get_xticklabels(), colors):
        xt.set_color(c)

    ax.legend_.remove()
    ax.text(3.5, 100, glabels[0], va='top', ha='left',
            fontsize=15, fontweight='bold', color=gcolors[0])
    ax.text(3.5, 80, glabels[1], va='bottom', ha='left',
            fontsize=15, fontweight='bold', color=gcolors[1])

    fig.tight_layout()
    if save_as:
        vut.save_it(fig, save_to, figname, save_as=save_as)


if __name__ == '__main__':
    cwd = path.dirname(__file__)
    make_fig(
        data_path=path.join(path.dirname(__file__), '../data/long3.pkl'),
        figname='figure1b',
        save_as='svg',
        save_to=path.join(path.dirname(__file__), '../figures')
    )
