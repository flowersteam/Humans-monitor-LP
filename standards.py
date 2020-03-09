class RAWix(object):
    def __init__(self, include_group=True):
        self.cols = ['sid', 'trial', 'cond', 'stage', 'blkt', 'fam', 'd1', 'd2', 'cat', 'food', 'cor', 'switch', 'rt']
        if include_group:
            self.cols.insert(0, 'group')
        self.ix = self.cols.index


class RAWXix(object):
    def __init__(self, include_group=True):
        self.cols = [
            'sid', 'cond',
            'q1m1', 'q2m1', 'q3m1', 'q4m1', 'q5m1', 'q6m1', 'q7m1',
            'q1m2', 'q2m2', 'q3m2', 'q4m2', 'q5m2', 'q6m2', 'q7m2',
            'q1m3', 'q2m3', 'q3m3', 'q4m3', 'q5m3', 'q6m3', 'q7m3',
            'q1m4', 'q2m4', 'q3m4', 'q4m4', 'q5m4', 'q6m4', 'q7m4'
                     ]
        if include_group:
            self.cols.insert(0, 'group')
        self.ix = self.cols.index

    def insert_col(self, where, what):
        self.cols.insert(where, what)


class SURix(object):
    # LP1 = first 5 / last 5 (pc)
    # LP2 = first 5 - last 5 (pc)
    # LP3 = average increase in pc across consequtive trials
    # LP4 = b1 of CORRECT = b0 + b1*TRIAL
    # LP5 = first 5 / last 5 (nbt)
    def __init__(self):
        self.cols = ['sid', 'group', 'cond', 'task', 'pc', 'lrn',
                     'lp1', 'lp2', 'lp3', 'lp4', 'lp5',
                     'pc_rank', 'lrn_rank3', 'lrn_rank4',
                     'choice']
        self.ix = self.cols.index
