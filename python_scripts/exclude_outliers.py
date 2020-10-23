import pandas as pd
import numpy as np
from os import path


# Set outlier criteria
av_crit = 100   # allocation variance critical value
rb_crit = 2     # choice bias critical value
activities = ['A1', 'A2', 'A3', 'A4']

# Define a response bias function
def rbf(x):
    _, response_counts = np.unique(x.response, return_counts=True)
    return np.max(response_counts) / np.sum(response_counts)


if __name__ == '__main__':
    # Open combined data file
    data_path = path.join(path.dirname(__file__), '../data')
    df = pd.read_csv(path.join(data_path, 'combined_main.csv'), index_col=None).set_index('sid')

    # Initialize columns to record values of interest
    df['alloc_var'], df['resp_bias'] = 0, 0

    # Calculate values of interest
    for sid, sdf in df.groupby(by='sid'):
        # Allocation variance
        counts = [sum(sdf.activity == i) for i in activities]
        allocation_variance = np.std(counts)
        df.loc[sid, 'alloc_var'] = allocation_variance

        # Response bias
        response_bias = sdf.groupby('family').apply(rbf).mean()
        df.loc[sid, 'resp_bias'] = response_bias

    # Detect high allocation variance and response bias
    df_ = df.reset_index().groupby('sid').head(1).reset_index()
    df_['high_av'] = df_.alloc_var >= av_crit
    df_['high_rb'] = np.logical_and(df_.resp_bias > df_.resp_bias.mean() + rb_crit * df_.resp_bias.std(), ~df_.high_av)

    print(df_.groupby(by='group')[['high_av', 'high_rb']].sum().astype(int))
    print('\nFound {} outliers\n'.format(np.logical_or(df_.high_av, df_.high_rb).sum()))

    # Exclude outliers
    outlier = df_.loc[df_.high_av | df_.high_rb, 'sid']
    df = df.loc[~df.index.isin(outlier), :]
    print(df.reset_index().groupby(by='group')['sid'].nunique())

    # Save data
    df.reset_index().to_csv(path.join(data_path, 'clean_data.csv'), index=False)
