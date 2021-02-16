from IPython.display import display
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import warnings
import contextlib

import vis_utils as vut
from supplementary.simple_choice_model import hits_gen
import loc_utils as lut
import ipywidgets as wid
import loc_utils as lut
from standards import *

# main_path = 'supplementary/simple_choice_model/data/main_data.pkl' 
# ntm_path = 'supplementary/simple_choice_model/data/ntm_data.pkl'
r = RAWix()
colors = ['#43799d', '#cc5b46', '#ffbb00', '#71bc78', '#43799d', '#cc5b46', '#ffbb00', '#71bc78']
tlabels = {
        1: '1D',
        2: 'I1D',
        3: '2D',
        4: 'R'}
pd.options.display.float_format = '{:.5f}'.format
weights = np.array([1/6,2/6,3/6])
# =============================================================

def dummy(n, i):
    dummy = np.zeros(n)
    dummy[i] = 1
    return dummy


def utility(x, alpha, beta, gamma, theta):
    ''' Utility function
    x must be a row (or an array of rows) 
    containing 4 values: LP, PC, I, and TR '''
    return np.sum(x * np.array([alpha, beta, gamma, theta]), axis=1)


def softmax(U, t=1, inv=True):
    if inv:
        return np.exp(U * t) / np.sum(np.exp(U * t))
    return np.exp(U / t) / np.sum(np.exp(U / t))


def choose(a):
    cum_probs = np.cumsum(a, axis=0)
    rand = np.random.rand(a.shape[-1])
    return (rand<cum_probs).argmax()


@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def rand_params(bounds):
    return np.array([np.random.uniform(l, u) if bounds else np.random.rand() for l, u in bounds])


def get_sub_data(sid=None, grp=None, ntm=None):
    arrdata = lut.unpickle(main_path)['main'][:, [r.ix('group'),r.ix('sid'),r.ix('trial'),r.ix('cat'),r.ix('cor'),r.ix('blkt')]]
    df_dict = {}
    for i, (col, dt) in enumerate(zip('grp,sid,trial,tid,cor,blkt'.split(','), [int,int,int,int,int,int])):
        df_dict[col] = pd.Series(arrdata[:, i], dtype=dt)
    df = pd.DataFrame(df_dict)

    ntm_df = lut.unpickle(ntm_path)[['sid','ntm']].groupby('sid').head(1)
    df = df.merge(ntm_df, on='sid').drop_duplicates().astype(int)
    del ntm_df, arrdata
    df = df.loc[df.trial<=60]
    if ntm: df = df.loc[df.ntm==ntm, :]
    if grp: df = df.loc[df.grp==grp, :]
    if sid: 
        df = df.loc[df.sid==int(sid), :]
    else:
        sid = np.random.choice(df.sid.unique())
        df = df.loc[df.sid==sid, :]

    df = df.loc[:, ['blkt','tid','cor']].pivot(index='blkt', columns='tid', values='cor')
    return df, sid


def get_sub_choices(sid):
    data = lut.unpickle(main_path)['main'][:, [r.ix('sid'),r.ix('trial'),r.ix('cat')]]
    filt = (data[:, 0]==sid) & (data[:, 1]>60) & (data[:, 1]<=310)
    data = data[filt, -1]-1
    return data.astype(int)


def get_multiple_sids(sids):
    arrdata = lut.unpickle(main_path)['main'][:, [r.ix('sid'),r.ix('trial'),r.ix('cat'),r.ix('cor')]]
    df_dict = {}
    for i, (col, dt) in enumerate(zip('sid,trial,tid,cor'.split(','), [int,int,int,int])):
        df_dict[col] = pd.Series(arrdata[:, i], dtype=dt)
    df = pd.DataFrame(df_dict)
    df = df.loc[df.trial<=60]

    arr = np.zeros([np.shape(sids)[0], 15, 4])
    for i, sid in enumerate(sids):
        for j, tid in enumerate([1,2,3,4]):
            arr[i, :, j] = df.loc[(df.sid==sid) & (df.tid==tid), 'cor'].values

    return arr


def prepare_fit_data(save_as=''):
    size0, size1, overlap = 10, 9, 4
    cols = ['sid','grp','stage','trial','t0','loc_p1','loc_p2','loc_p3','loc_p4','loc_pc1','loc_pc2','loc_pc3','loc_pc4','cor']
    df = lut.unpickle('supplementary/simple_choice_model/data/trials_data_w15.pkl')[cols]
    df = df.loc[df.trial<=310, :]

    sids, grps, ntms, trials, glps, gpcs, gins, gchs = [], [], [], [], [], [], [], []
    for i, sdf in df.groupby('sid'):
        crit_pval = sdf.loc[:, 'loc_p1':'loc_p3'] <= .01
        crit_pc = sdf.loc[:, 'loc_pc1':'loc_pc3'] > .5
        learned = crit_pval.values & crit_pc.values
        ntm = np.any(learned, axis=0).sum()
        sid = sdf.loc[:, 'sid'].values[0]
        grp = sdf.loc[:, 'grp'].values[0]

        mem = np.full([size0+size1-overlap, 4], np.nan)
        mem[:, 0] = sdf.loc[(sdf.t0==1) & (sdf.stage==0), 'cor']
        mem[:, 1] = sdf.loc[(sdf.t0==2) & (sdf.stage==0), 'cor']
        mem[:, 2] = sdf.loc[(sdf.t0==3) & (sdf.stage==0), 'cor']
        mem[:, 3] = sdf.loc[(sdf.t0==4) & (sdf.stage==0), 'cor']

        # Data of the 1st free-play trial
        lps = [np.abs(mem[:-size0, :].mean(axis=0) - mem[-size1:, :].mean(axis=0))] 
        pcs = [np.mean(mem, axis=0)]
        chs = [np.eye(4)[sdf.t0.values[61]-1, :]]

        x = np.stack([lps[0], pcs[0]], axis=0).T
        choices = sdf.t0.values[62:]
        cor = sdf.cor.values[61:-1]
        for tid, outcome in zip(choices, cor):
            j = tid - 1 # choice index

            # Update hits memory
            mem[:-1, j] = mem[1:, j]
            mem[-1, j] = outcome

            # Update expected reward (PC)
            pc_vect = np.mean(mem, axis=0)
            x[j, 1] = pc_vect[j] # PC

            # Update LP
            lp_vect = np.abs(mem[:-size0, :].mean(axis=0) - mem[-size1:, :].mean(axis=0))
            x[j, 0] = lp_vect[j] # LP

            # ========== Record data ============
            pcs.append(pc_vect)
            lps.append(lp_vect)
            chs.append(np.eye(4)[j, :])

        glps.append(np.stack(lps, axis=0))
        gpcs.append(np.stack(pcs, axis=0))
        gchs.append(np.stack(chs))
        gins.append(np.zeros_like(lps))
        gins[-1][1:] = chs[:-1]

        sids.append(np.ones([len(lps), 1]) * sid)
        grps.append(np.ones([len(lps), 1]) * grp)
        ntms.append(np.ones([len(lps), 1]) * ntm)
        trials.append((np.arange(len(lps))+1).reshape([-1, 1]))

    cols = []
    for col in (sids, grps, ntms, trials, glps, gpcs, gins, gchs):
        # print(np.concatenate(col, axis=0).shape)
        cols.append(np.concatenate(col, axis=0))

    data = np.concatenate(cols, axis=1)
    colnames = 'sid,grp,ntm,trial,lp1,lp2,lp3,lp4,pc1,pc2,pc3,pc4,in1,in2,in3,in4,ch1,ch2,ch3,ch4'.split(',')
    df = pd.DataFrame(data, columns = colnames)
    for colname in 'sid,grp,ntm,trial,in1,in2,in3,in4,ch1,ch2,ch3,ch4'.split(','):
        df.loc[:, colname] = df.loc[:, colname].astype(int)
    
    if save_as: lut.dopickle(save_as, df)


def evaluate(sdata, choices, crit_pc=.5, crit_pval=.01):
    pcs = np.stack(sdata, axis=0)
    pvals = lut.p_val(15, pcs*15, .5)
    weighted_scores = np.sum(pcs[-1, :-1]*np.array([1,2,3])/6)
    crit_pc = pcs[:, :-1] > crit_pc
    crit_pval = pvals[:, :-1] <= crit_pval
    learned = crit_pc & crit_pval

    lps = np.argmax(learned, axis=0)
    ntm = np.any(learned, axis=0).sum()

    pcs = pcs[1:, :]
    _max = pd.Series(pcs.max(axis=1)).rolling(pcs.shape[0], min_periods=1).max()
    _min = pd.Series(pcs.min(axis=1)).rolling(pcs.shape[0], min_periods=1).min()
    _pc = pcs[np.arange(pcs.shape[0]), choices.astype(int)]
    _sc = 1 - (_pc-_min)/(_max-_min)

    _sorted_lep_bounds = np.sort(np.unique([0]+lps.tolist()+[3]))
    _lep_intervals = pd.IntervalIndex.from_arrays(_sorted_lep_bounds[:-1], _sorted_lep_bounds[1:], closed='right')
    sc_lep = _sc.groupby(pd.cut(np.arange(pcs.shape[0]), _lep_intervals)).mean().mean()

    return weighted_scores, sc_lep


def simple_simulation(init_state, win1, win2, N, hits, alpha, beta, gamma, theta, tau, inverse_temp=True):
        counter = np.array([1, 1, 1, 1])
        timerel = counter.copy()*15
        mem = init_state.copy()
        init_pc = np.mean(mem, axis=0)
        init_lp = np.abs(mem[:win1, :].mean(axis=0) - mem[win2:, :].mean(axis=0))
        init_in = np.zeros_like(init_pc)
        init_tr = np.zeros_like(init_pc)
        choices = np.zeros(N)
        x = np.stack([init_lp, init_pc, init_in, init_tr], axis=0).T
        for trial in range(N):
            # Compute utility based on state x, choose the next task based on utility, 
            # and get feedback by playing the task
            U = utility(x, alpha=alpha, beta=beta, gamma=gamma, theta=theta)
            i = np.random.choice([0,1,2,3], size=1, p=softmax(U, tau, inverse_temp))[0]
            hit = hits[counter[i]-1, i]
            counter[i] += 1
            timerel[i] += 1

            # ========== Update memory ==========
            # 1. Update last choice
            x[:, 2] = 0 
            x[i, 2] = 1

            # 2. Update hits memory
            mem[:-1, i] = mem[1:, i]
            mem[-1, i] = hit

            # 3. Update expected reward (PC)
            pc_vect = np.mean(mem, axis=0)
            x[i, 1] = pc_vect[i] # PC

            # 4. Update LP
            lp_vect = np.abs(mem[:win1, :].mean(axis=0) - mem[win2:, :].mean(axis=0))
            x[i, 0] = lp_vect[i] # LP

            # 5. Update TR
            tr_vect = timerel / np.sum(timerel)
            x[:, 3] = tr_vect

            choices[trial] = i
        return choices
        # return counter/np.sum(counter), pcs, lps, choices, hits, util


def neg_log_likelihood(params, *args):
    a, b, c, d, t = params
    LP, PC, I, TR, choices = args
    U = a*LP + b*PC + c*I + d*TR
    P = (np.exp(U * t).T / np.sum(np.exp(U * t), axis=1)).T
    logP = np.log(P[choices.astype(bool)])
    logL = np.sum(logP, axis=0)
    return -logL


class Simulator():
    def __init__(self, nb_trials, hits_generator, controls=True, live=False, seed=1, max_rolls=30,
                 alpha=[-1,1], beta=[-1,1], gamma=[0,1], theta=[-1,0], tau=[1,10]):
        self._first = True
        self.nb_trials = nb_trials
        self.max_rolls = max_rolls
        
        self.win1 = 10
        self.win2 = 9
        self.overlap = 4
        self.memcap = self.win1 + self.win2 - self.overlap
        self.tids = np.array([0,1,2,3])

        # WIDGETS
        self.seed = wid.BoundedIntText(min=1, max=100000, value=seed, description='Seed', layout=wid.Layout(width='15%'), continuous_update=live)
        self.rolls = wid.BoundedIntText(min=1, max=self.max_rolls, value=1, description='N runs', layout=wid.Layout(width='15%'), continuous_update=live)

        self.alpha = wid.FloatSlider(min=alpha[0], max=alpha[1], value=np.random.uniform(*alpha), step=.01, description='alpha (LP)', continuous_update=live, layout=wid.Layout(width='80%'))
        self.beta = wid.FloatSlider(min=beta[0], max=beta[1], value=np.random.uniform(*beta), step=.01, description='beta (PC)', continuous_update=live, layout=wid.Layout(width='80%'))
        self.gamma = wid.FloatSlider(min=gamma[0], max=gamma[1], value=np.random.uniform(*gamma), step=.01, description='gamma (I)', continuous_update=live, layout=wid.Layout(width='80%'))
        self.theta = wid.FloatSlider(min=theta[0], max=theta[1], value=np.random.uniform(*theta), step=.01, description='theta (TR)', continuous_update=live, layout=wid.Layout(width='80%'))
        self.tau = wid.FloatSlider(min=tau[0], max=tau[1], value=np.random.uniform(*tau), step=.01, description='tau (inv. temp)', layout=wid.Layout(width='80%'), continuous_update=live)
        self.trial = wid.IntSlider(min=0, max=nb_trials-1, value=nb_trials-1, description='trial', continuous_update=live, layout=wid.Layout(width='80%'))

        self.rid = wid.IntSlider(min=1, max=1, value=1, description='Run #', continuous_update=live)
        self.rolls.observe(self.rolls_change, 'value')
        self.grp_picker = wid.Dropdown(options=[('All', None), ('Free', 0), ('Strategic', 1)], value=None, description='Group: ', layout=wid.Layout(width='20%'))
        self.ntm_picker = wid.Dropdown(options=[('All', None), ('1', 1), ('2', 2), ('3', 3)],   value=None, description='NTM: ', layout=wid.Layout(width='20%'))
        self.sid_picker = wid.Text(value='', placeholder='ID or blank', description='Subject ID', layout=wid.Layout(width='20%'))
        self.update_button = wid.Button(description='Update initial state', button_style='info')
        self.update_button.on_click(lambda x: self.update_init_state())
        
        self.sim_button = wid.Button(description='Simulate', button_style='success', layout=wid.Layout(width='120px'))
        self.sim_button.on_click(lambda x: self.run_sim())

        self.fit_button = wid.Button(description='Fit', button_style='warning', layout=wid.Layout(width='100px'))
        self.fit_button.on_click(lambda x: self.fit_params())

        self.out = wid.Output()
        self.out2 = wid.Output()
        
        self.bounds = [(p.min, p.max) for p in [self.alpha, self.beta, self.gamma, self.theta, self.tau]]
        self.hits_generator = hits_generator
        self.init_state = None
        self.controls = controls

        if controls:
            self.fig1 = plt.figure('sim^1', figsize=[8, 4.5])

            self.ax1 = vut.pretty(self.fig1.add_subplot(221))
            self.ax1.set_ylim(-.5, 3.5)
            self.ax1.set_yticks([0,1,2,3])
            self.ax1.set_yticklabels([tlabels[i] for i in [4,3,2,1]])
            self.ax1.set_xticks(list(range(15)))
            self.ax1.set_xticklabels(list(range(15)))
            self.ax1.imshow(np.zeros([4,15]), cmap='binary')
            
            self.ax2 = vut.pretty(self.fig1.add_subplot(222))
            self.ax2.set_xticks([0,1,2,3])
            self.ax2.set_xticklabels([tlabels[i] for i in [1,2,3,4]])
            self.ax2.set_ylim(0, 1)
            self.ax2.set_xlim(-.5, 3.5)
            
            self.ax3 = vut.pretty(self.fig1.add_subplot(212))
            self.ax3.set_ylim(-4, 1)
            self.ax3.set_yticks([-3,-2,-1,0])
            self.ax3.set_yticklabels([tlabels[i] for i in [4,3,2,1]])
            self.traj = None
            self.simhits = None
            self.simmisses = None
            self.vertline = None
            self.simbars = None
            self.truebars = None

            # display controls before Axes4
            self.controls_on()

            # self.fig2 = plt.figure('sim^2', figsize=[6, 3])
            # self.ax4 = vut.pretty(self.fig2.add_subplot(121))
            # self.ax4.set_xlim(-.02, 1.02)
            # self.ax4.set_ylim(0.6, 1.02)
            # self.ax4.set_ylabel('Weighted score')
            # self.ax4.set_xlabel('Self-challenge')
            # self.ax4.set_title('Learning outcomes')

            # self.ax5 = vut.pretty(self.fig2.add_subplot(122))
            # self.ax5.set_xticks([0,1,2,3])
            # self.ax5.set_xticklabels([tlabels[i] for i in [1,2,3,4]]) 
            # self.ax5.set_ylabel('% selection')
            # self.ax5.set_xlabel('Task ID')
            # self.ax5.set_ylim(0, 1)
            # self.ax4.set_title('Mean task selection')
            # self.fig2.tight_layout()

    def controls_on(self):
        display(self.out2)
        display(wid.HBox([self.update_button, self.sid_picker, self.grp_picker, self.ntm_picker]), 
                wid.HBox([self.sim_button, self.fit_button, self.seed, self.rolls , self.rid]))
        display(wid.VBox([self.alpha, self.beta, self.gamma, self.theta, self.tau, self.trial, self.out]))

    def update_init_state(self):
        sid, grp, ntm = self.sid_picker.value, self.grp_picker.value, self.ntm_picker.value
        if sid=='': sid=None
        new_init_state, new_init_state_id = get_sub_data(sid, grp, ntm)
        if new_init_state.empty:
            self.out.clear_output(wait=True)
            with self.out: print('Subject {} not found in (GRP=\'{}\' & NTM=\'{}\'). Init state not updated, try another ID'.format(sid, grp if grp is not None else 'F|S', ntm if ntm else '1|2|3'))
        else: 
            self.init_state, self.init_state_id = new_init_state, new_init_state_id
            self.sid_picker.placeholder = str(self.init_state_id)
            self.sid_picker.value = ''
            if self.controls:
                choices = get_sub_choices(int(self.sid_picker.placeholder))
                self.ax1.imshow(np.flipud(self.init_state.T.values.copy()), cmap='binary')
                self.ax1.set_title('SID: {}'.format(self.init_state_id))
                if self.traj: self.traj = self.traj.remove()
                self.traj, = self.ax3.plot(np.arange(choices.size), -choices, c='k', lw=2)

                if self.truebars: 
                    for bar in self.truebars: bar.remove()

                self.truebars = self.ax2.bar(np.arange(self.init_state.shape[1])-.15, 
                                  np.eye(4)[choices].mean(axis=0), width=.30,
                                  color='k', edgecolor='white')
                 
    def simulate(self, alpha, beta, gamma, theta, tau):
        counter = np.array([1, 1, 1, 1])
        timerel = counter.copy()*15
        mem = self.init_state.values.copy()[-self.memcap:, :]
        init_pc = np.mean(mem, axis=0)
        init_lp = np.abs(mem[:-self.win1, :].mean(axis=0) - mem[-self.win2:, :].mean(axis=0))
        init_in = np.zeros_like(init_pc)
        init_tr = timerel / np.sum(timerel)
        lps, pcs, trs = [init_lp], [init_pc], [init_tr]
        util = []
        choices, hits = np.zeros(self.nb_trials), np.zeros(self.nb_trials)

        x = np.stack([init_lp, init_pc, init_in, init_tr], axis=0).T
        for trial in range(self.nb_trials):
            # Compute utility based on state x, choose the next task based on utility, 
            # and get feedback by playing the task
            U = utility(x, alpha, beta, gamma, theta)
            i = np.random.choice(self.tids, size=1, p=softmax(U, tau))[0]
            counter[i] += 1
            timerel[i] += 1
            hit = self.hits_generator.generate(counter[i], i+1)

            # ========== Update memory ==========
            # 1. Update last choice
            x[:, 2] = 0 
            x[i, 2] = 1

            # 2. Update hits memory
            mem[:-1, i] = mem[1:, i]
            mem[-1, i] = hit

            # 3. Update expected reward (PC)
            pc_vect = np.mean(mem, axis=0)
            x[i, 1] = pc_vect[i] # PC

            # 4. Update LP
            lp_vect = np.abs(mem[:-self.win1, :].mean(axis=0) - mem[-self.win2:, :].mean(axis=0))
            x[i, 0] = lp_vect[i] # LP

            # 5. Update TR
            tr_vect = timerel / np.sum(timerel)
            x[:, 3] = tr_vect

            # ========== Record data ============
            pcs.append(pc_vect)
            lps.append(lp_vect)
            trs.append(tr_vect)
            util.append(U)
            choices[trial] = i
            hits[trial] = hit
                
        return counter/np.sum(counter), pcs, lps, trs, choices, hits, util

    def plot_sim(self, t, alpha, beta, gamma, theta, tau, seed, N, rid):
        with temp_seed(seed):
            tot_list, pc_list, lp_list, tr_list, choices_list, hits_list, util_list = [], [], [], [], [], [], []
            data_list = [tot_list, pc_list, lp_list, tr_list, choices_list, hits_list, util_list]
            for i in range(N):
                with self.out:
                    tot, pc, lp, tr, choices, hits, util = self.simulate(alpha, beta, gamma, theta, tau)
                    for lst, data in zip(data_list, [tot, pc, lp, tr, choices, hits, util]): lst.append(data)

            # while len(self.ax2.findobj(match=mpl.patches.Rectangle)) > 1: 
            #     self.ax2.findobj(match=mpl.patches.Rectangle)[0].remove()
            # rects = self.ax2.bar(np.arange(self.init_state.shape[1]), tot_list[rid-1], color='white', edgecolor='k')

            if self.simhits: self.simhits = self.simhits.remove()
            if self.simmisses: self.simmisses = self.simmisses.remove()
            if self.vertline: self.vertline = self.vertline.remove()
            self.vertline = self.ax3.axvline(t, color='#F6D700')

            inds = np.arange(choices_list[rid-1].size)
            hmask = hits_list[rid-1].astype(bool)
            self.simhits, = self.ax3.plot(inds[hmask], -choices_list[rid-1][hmask], 
                c='green', marker='|', ls='', ms=12, mew=2, alpha=.6)
            self.simmisses, = self.ax3.plot(inds[~hmask], -choices_list[rid-1][~hmask], 
                c='red', marker='|', ls='', ms=12, mew=2, alpha=.6)

            self.out.clear_output(wait=True)
            with self.out: 
                choices_binary = np.zeros([choices_list[rid-1].size, 4])
                choices_binary[np.arange(choices_list[rid-1].size), choices_list[rid-1].astype(int)] = 1
                NA = ['N/A','N/A','N/A','N/A']
                nohit = [False, False, False, False]
                out = pd.DataFrame({
                    'tid': [tlabels[tid] for tid in [1,2,3,4]], 
                    'LP(t-)': lp_list[rid-1][t], 
                    'PC(t-)': pc_list[rid-1][t], 
                    'I(t-)': choices_binary[t-1, :].astype(int) if t else np.zeros(4).astype(int),
                    'TR(t-)': tr_list[rid-1][t],
                    'utility(t)': util_list[rid-1][t],
                    'p(t)': softmax(util_list[rid-1][t], tau),
                    'choice(t)': choices_binary[t, :].astype(bool),
                    'hit(t)': choices_binary[t, :].astype(bool) if hits_list[rid-1][t] else nohit,
                    'LP(t+)': lp_list[rid-1][t+1], 
                    'PC(t+)': pc_list[rid-1][t+1], 
                }).set_index('tid')
                display(out)


            # while len(self.ax4.findobj(match=mpl.collections.PathCollection)) > 0: 
            #     self.ax4.findobj(match=mpl.collections.PathCollection)[0].remove()

            # scores = np.array([evaluate(pcs, choices) for pcs, choices in zip(pc_list, choices_list)])

            # self.ax4.scatter(scores[:, 1], scores[:, 0], edgecolor='darkgray', facecolor='w', alpha=.6)
            # self.ax4.scatter(scores[rid-1, 1], scores[rid-1, 0], marker='s', facecolor='w', edgecolor='darkgray')
            # self.ax4.scatter(scores[:, 1].mean(), scores[:, 0].mean(), facecolor='k', edgecolor='k')

            if self.simbars: 
                for bar in self.simbars: bar.remove()
            self.simbars = self.ax2.bar(np.arange(self.init_state.shape[1])+.15, 
                                  np.stack(tot_list, axis=0).mean(axis=0), width=.30,
                                  color='#4CAF50', edgecolor='white')

            self.out2.clear_output(wait=True)
            with self.out2:
                data = self.get_fit_data()
                params = (alpha, beta, gamma, theta, tau)
                loss = neg_log_likelihood(params, *data)
                print('Neg. log likelihood =', round(loss,4))

            return out
            
    def run_sim(self):
        if self._first:
            self._first = False
            self.sim = wid.interactive(self.plot_sim,
                                        t = self.trial,
                                        alpha = self.alpha, 
                                        beta = self.beta, 
                                        gamma = self.gamma,
                                        theta = self.theta,
                                        tau = self.tau,
                                        seed = self.seed,
                                        N = self.rolls,
                                        rid = self.rid)
            self.sim.update()
        else:
            self.sim.update()

    def rolls_change(self, change):
        self.rid.max = change.new

    def fit_params(self):
        data = self.get_fit_data()
        params = (self.alpha, self.beta, self.gamma, self.theta, self.tau)
        bounds = [(p.min, p.max) for p in params]

        fit_vals = np.zeros([self.rolls.value, len(params)])
        losses = np.zeros([self.rolls.value])
        # for i in range(self.rolls.value):
        guess = rand_params(bounds)
        x, f, d = sp.optimize.fmin_l_bfgs_b(func=neg_log_likelihood, x0=guess, args=data,
                                            approx_grad=True, disp=True, bounds=bounds)
            # fit_vals[i, :] = x
            # losses[i] = f
        # fit_vals = fit_vals.mean(axis=0)
        fit_vals = fit_vals[losses.argmin(), :]
        for fit_val, param in zip(x, params):
            param.value = fit_val

    def get_fit_data(self):
        df = lut.unpickle('supplementary/simple_choice_model/data/fit_data.pkl')
        df = df.loc[df.sid==int(self.sid_picker.placeholder)]
        lps = df.loc[:, 'lp1':'lp4'].values[1:, :]
        pcs = df.loc[:, 'pc1':'pc4'].values[1:, :]
        ins = df.loc[:, 'in1':'in4'].values[1:, :]
        chs = df.loc[:, 'ch1':'ch4'].values[1:, :]
        time_alloc = (df.loc[:, 'ch1':'ch4'].values[1:, :].cumsum(axis=0) + 15)
        trs = (time_alloc.T / time_alloc.sum(axis=1)).T
        return lps, pcs, ins, trs, chs
