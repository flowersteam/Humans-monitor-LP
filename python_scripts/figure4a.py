import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from os import path

from python_scripts.utils import vis_utils as vut


def make_fig(data_path, figname, save_to, save_as=''):
    baseline = 693.1471805599454
    ic = 'aic'
    df = pd.read_csv(data_path)
    fdf = pd.read_csv(data_path)[['form', 'loss', ic]]
    fdf['nvars'] = 6 - df.loc[:, 'tau':'relt_coef'].isnull().sum(axis=1)
    fdf.loc[:, ic] = 2 * fdf.loss + 2 * fdf.nvars
    fdf = fdf[~fdf.form.str.contains('dLP')]
    fdf = fdf[~fdf.form.str.contains('mLP')]
    fdf = fdf[~fdf.form.str.contains('RELT')]
    fdf.replace(['rLP', 'PC + rLP'], ['LP', 'PC + LP'], inplace=True)
    byform = fdf.set_index('form')

    fdf = fdf.loc[:, ('form', 'aic')].groupby(['form']).agg(['mean', 'std'])
    fdf = fdf.sort_values(by=(ic, 'mean'), ascending=True)
    print(fdf)

    fig = plt.figure(figname, figsize=(5, 3.5))
    ax = vut.pretty(fig.add_subplot(111), 'x')

    fdf = fdf.sort_values(by=(ic, 'mean'), ascending=False)
    sns.violinplot(x=ic, y='form', data=byform.reset_index(), ax=ax,
                   color='lightgray', linewidth=1.5, order=fdf.index)
    for v in ax.collections: v.set_edgecolor('w')

    ax.axvline(baseline, ls='--', zorder=3, color='red',
               label='Random model')
    ax.set_ylabel('Model', fontsize=14)
    ax.set_xlabel('AIC', fontsize=14)
    ax.set_xlim([200, 800])
    ax.legend()

    print('Baseline AIC = {}'.format(baseline))
    fig.tight_layout()
    if save_as:
        vut.save_it(fig, save_to, figname=figname, save_as=save_as, compress=False, dpi=100)


if __name__ == '__main__':
    data_path = path.join(path.dirname(__file__), '../data/choiceModelComparison_PC_dLP_rLP_mLP_RELT_300seeds.csv')
    save_path = path.join(path.dirname(__file__), '../figures')

    make_fig(
        data_path=data_path,
        figname='figure4a',
        save_to=save_path,
        save_as='svg'
    )