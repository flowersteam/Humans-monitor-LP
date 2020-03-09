import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import rankdata, pearsonr
import scipy.stats as scs
from scipy.special import comb
import statsmodels.api as sm

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


def smooth(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


def pretty_scatter(x, y ,ax, xlim=None, ylim=None, xlabel=None, ylabel=None, dot_col='k', line_col='r', textpos='top-right'):
    x, y = x[~np.isnan(y)], y[~np.isnan(y)]
    x, y = x[~np.isnan(x)], y[~np.isnan(x)]
    
    rho = pearsonr(x,y)
    
    va, ha = textpos.split('-')
    coords = {'top': .9, 'bottom': .1, 'left': .1, 'right': .9}
    
    ax.text(coords[ha], coords[va], 'r = {:.3f}\np = {:.4f}'.format(rho[0], rho[1]), 
     ha=ha, va=va, transform=ax.transAxes, color='k' if rho[1] < .025 else 'gray')

    ax.scatter(x, y, alpha=.4,
           marker='o', s=5, color=dot_col)
    
    if line_col:
        x = sm.add_constant(x)
        model = sm.OLS(y, x)
        results = model.fit()
        linex, liney = xlim
        ax.plot(np.array([linex, liney]), 
        results.params[0] + np.array([linex, liney])*results.params[1],
        lw=1, c=line_col)

    if xlim: ax.set_ylim(ylim)
    if ylim: ax.set_xlim(xlim)
        
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    
    
    despine(ax, ['top', 'right'])
    ax.grid(True)


def very_pretty_scatter(x, y, ax, groups, gcolors, glabels, 
    m='o', xlim=None, ylim=None, xlabel=None, ylabel=None, 
    textpos='top-right', cutoff=.025):
    groups = groups.astype(int)
    
    r, pval = pearsonr(x,y)
    x_ = sm.add_constant(x)
    model = sm.OLS(y, x_)
    results = model.fit()
    linex, liney = xlim
    label = 'F and S'
    res = '\nr={:.3f}, p={:.3f}'.format(r, pval)
    sign = ' **' if pval < .01 else ' *' if pval < .05 else ''
    lstyle = '-' if pval < .01 else (0, (5, 5)) if pval < .05 else (0, (5, 10))

    ax.plot(np.array([linex, liney]), 
        results.params[0] + np.array([linex, liney])*results.params[1],
        lw=1, c='k', label=label+sign+res, ls=lstyle)
        
    for grp in np.unique(groups):
        r, pval = pearsonr(x[groups==grp], y[groups==grp])
        ax.scatter(x[groups==grp], y[groups==grp],  alpha=.4,
               marker=m, s=5, color=gcolors[grp])
        x_ = sm.add_constant(x[groups==grp])
        model = sm.OLS(y[groups==grp], x_)
        results = model.fit()
        linex, liney = xlim
        lstyle = '-' if pval < .01 else (0, (5, 5)) if pval < .05 else (0, (5, 10))

        label = glabels[grp]
        res = '\nr={:.3f}, p={:.3f}'.format(r, pval)
        sign = ' **' if pval < .01 else ' *' if pval < .05 else ''

        ax.plot(np.array([linex, liney]), 
            results.params[0] + np.array([linex, liney])*results.params[1],
            lw=1, ls=lstyle, c=gcolors[grp],
            label=label+sign+res)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True)
    
    
def pad_lims(a, scale=.05):
    mn, mx = np.min(a), np.max(a)
    rng = np.abs(mx - mn)
    return (mn-rng*scale, mx+rng*scale)


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


def rand_jitter(arr):
    stdev = .01*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev


def color_legend(legend, fontweight='bold'):
    for i, (line, text) in enumerate(zip(legend.get_lines(), legend.get_texts())):
        text.set_color(line.get_color())
        text.set_fontweight(fontweight)
    

def label_diff(ax, i, j, text, X, Y):
    x = (X[i]+X[j])/2
    y = 1.1*max(Y[i], Y[j])
    dx = abs(X[i]-X[j])

    props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':20,'shrinkB':20,'linewidth':2}
    ax.annotate(text, xy=(X[i],y+7), zorder=10)
    ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
    
    
def metric(width, height, unit='mm'):
    O = {'mm': .001, 'cm': .01, 'm': 1}
    C = 39.37008
    return ((width*O[unit])*C, (height*O[unit])*C)
    