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
    

def ghostify(ax):
    ax.tick_params(left=False, labelleft=False, bottom=False, labelbottom=False)
    for spine in ax.spines.values(): spine.set_visible(False)
    
    
def metric(width, height, unit='mm'):
    O = {'mm': .001, 'cm': .01, 'm': 1}
    C = 39.37008
    return ((width*O[unit])*C, (height*O[unit])*C)


def strip_axes(ax, remove_spines=False):
    ax.tick_params(labelbottom=False, labelleft=False,
                   bottom=False, left=False)
    if remove_spines:
        for pos in 'left,right,top,bottom'.split(','):
            ax.spines[pos].set_visible(False)


gcolors = ['#008fd5', '#fc4f30']
gmarkers = ['s', 'o']
colors = ['#65C2A5', '#FC8D62', '#8DA0CB', '#E78AC3']
ncolors = ['#7BD8ED','#656CBE', '#4E008E']
glabels = {0: 'IG', 1: 'EG'}
fullglabels = {0: 'IG', 1: 'EG'}
tlabels = OrderedDict({
        1: 'A1',
        2: 'A2',
        3: 'A3',
        4: 'A4'})
