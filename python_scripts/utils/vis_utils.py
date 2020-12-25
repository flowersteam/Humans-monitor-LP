import os
import numpy as np
from collections import OrderedDict


def fd_bins(x):
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25

    binwidth = (2 * iqr / (len(x) ** (1 / 3)))
    nbins = np.ptp(x) / binwidth
    return np.round(nbins).astype(int)


def add_stds(ax, data, n=5, showmean=True):
    mean = np.mean(data)
    std = np.std(data)
    if showmean:
        ax.axvline(mean, c='r', lw=1.5)
    for i in range(n):
        if i % 2 == 0:
            ax.axvspan(mean + i * std, (mean + i * std) + std, alpha=0.6, color='lightgray')
            ax.axvspan(mean - i * std, (mean - i * std) - std, alpha=0.6, color='lightgray')


def add_labels(ax, title=None, x=None, y=None):
    if title:
        ax.set_title(title)
    if x:
        ax.set_xlabel(x)
    if y:
        ax.set_ylabel(y)


def despine(ax, which=['top','right']):
    if isinstance(which, str):
        which = [which]
    for spine in which:
        ax.spines[spine].set_visible(False)


def line_histogram(ax, data, bins, label, precision=None, lw=1, c=None, alpha=1, ls='-'):
    if precision:
        data = np.around(data, precision)
        bins = np.around(bins, precision)
    y, bin_edges = np.histogram(data, bins=bins, density=False, weights=np.zeros_like(data) + 1. / data.size)
    bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
    if c:
        ax.plot(bin_centers, y, '-', label=label, lw=lw, c=c, alpha=alpha, ls=ls)
    else:
        ax.plot(bin_centers, y, '-', label=label, lw=lw, alpha=alpha, ls=ls)
    ax.set_xticks(bins[:-1])
    ax.grid(axis='y', c='gray', ls='dotted')
    ax.grid(axis='x', c='gray', ls='dotted')
    despine(ax, ('top', 'right'))
    return y


def pretty(ax, gridlines='both'):
    despine(ax)
    ax.grid(True, c='gray', zorder=2, ls=':', axis=gridlines)
    return ax


def save_it(fig, savedir, figname, save_as='svg', dpi=300, compress=False):
    s = savedir+'/{}.{}'.format(figname, save_as)
    fig.savefig(s, format=save_as, dpi=dpi)
    if compress:
        os.system('scour -i {} -o {}'.format(s, s.replace('img', 'img_compressed')))
    print('SAVED TO: {}'.format(s))


def color_legend(legend, fontweight='bold'):
    for i, (line, text) in enumerate(zip(legend.get_lines(), legend.get_texts())):
        text.set_color(line.get_color())
        text.set_fontweight(fontweight)


def ghost(ax):
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ax.spines.values(): spine.set_visible(False)
    return ax

def add_subplot_label(ax, x, y, label, size, preview=False):
    ax.text(x, y, label, transform=ax.transAxes,
        size=size, weight='bold')
    if preview:
        ghost(ax)
        ax.set_facecolor('gray')
    else:
        ghost(ax)


def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


def strip_axes(ax, remove_spines=False):
    ax.tick_params(labelbottom=False, labelleft=False,
                   bottom=False, left=False)
    if remove_spines:
        for pos in 'left,right,top,bottom'.split(','):
            ax.spines[pos].set_visible(False)


gcolors = ['#008fd5', '#fc4f30']
gmarkers = ['s', 'o']
colors = ['#65C2A5', '#D4D669', '#8DA0CB', '#E78AC3']
ncolors = ['#7BD8ED','#656CBE', '#4E008E']
glabels = {0: 'IG', 1: 'EG'}
fullglabels = {0: 'IG', 1: 'EG'}
tlabels = OrderedDict({
        1: 'A1',
        2: 'A2',
        3: 'A3',
        4: 'A4'})
