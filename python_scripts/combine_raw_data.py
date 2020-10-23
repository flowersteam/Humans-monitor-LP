import pandas as pd
from os import path

# Find where data is stored
data_path = path.join(path.dirname(__file__), '../data')

if __name__ == '__main__':
    # Open main files and combine them
    df = pd.concat(
        (
            pd.read_csv(path.join(data_path, 'raw/ig_main.csv')),
            pd.read_csv(path.join(data_path, 'raw/eg_main.csv'))
        )
    )

    # Codify subject IDs
    df.loc[:, 'sid'] = df.sid.astype('category').cat.codes

    # Save combined data
    df.to_csv(path.join(data_path, 'combined_main.csv'), index=False)