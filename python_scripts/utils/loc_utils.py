import pandas as pd
import numpy as np
import pickle
import os


def may_be_make_dir(path):
    """
    If the directory is created between the `os.path.exists` and the `os.makedirs` calls, the `os.makedirs` will fail
    with an OSError.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def unpickle(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def dopickle(path, data):
    if os.path.isfile(path):

        overwrite = input('File {} exists. Overwrite? [y/n]\n>>> '.format(path))

        if overwrite.lower() == 'y':
            print('Overwriting data to {}'.format(path))
            with open(path, 'wb') as file:
                pickle.dump(data, file)
            print('Done saving.')
        else:
            print('Data not saved.')
            return

    else:
        may_be_make_dir(os.path.dirname(path))
        print('Saving data to {}'.format(path))
        with open(path, 'wb') as file:
            pickle.dump(data, file)
        print('Done saving.')


def get_sdf(data_path, sid, verbose=True):
    df = pd.read_csv(data_path, index_col='sid')
    if sid < 0:
        rsid = np.random.choice(df.index.tolist())
        sdf = df.loc[rsid, :]
    else:
        rsid = sid
        sdf = df.loc[sid, :]
    if verbose:
        print('Sampling sid: {}'.format(rsid))
    return sdf


def boolean_indexing(l, fillval=5):
    """
    Given a list of lists-likes of different lengths, output a rectangular array with shape[0] equal to the biggest list in v
    :param l: uneven list of lists
    :param fillval: value used to fill shorter list-likes with
    :return: rectangular array
    """
    lens = np.array([len(item) for item in l])
    mask = lens[:, None] > np.arange(lens.max())
    out = np.full(mask.shape, fillval)
    out[mask] = np.concatenate(l)
    return out


def onehotize(a):
    return (np.arange(a.max()) == a[..., None]-1).astype(int)
