
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots
from os.path import join
plt.style.use('science')

# FONTS
from font import SETTINGS
for key in SETTINGS:
    plt.rc(key, **SETTINGS[key])

COLOR = plt.rcParams['axes.prop_cycle']
COLOR = [v for elem in list(COLOR) for _, v in elem.items()]

CLR = {
    'dyt': COLOR[0],
    'dyisrusp': COLOR[1],
    'dytsp': COLOR[2],
    'dyisru': COLOR[4],
    'ln': COLOR[3],
}

LABEL = {
    'ln': 'LN',
    'dyt': 'DyT',
    'dyisrusp': r'DyISRU\textsuperscript{(sp)}',
    'dytsp': r'DyT\textsuperscript{(sp)}',
    'dyisru': 'DyISRU',
    'diff': 'DIFF',
}

def _save_figure(_save_as: str):
    if len(_save_as):
        assert _save_as.endswith('.pdf')
        save_path = join('figs', _save_as)
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f'> saved as {save_path}')

def plot(valloss, title=None, xmin=None, xmax=None, ymin=None, ymax=None, marker='o', additional=False, shade=None) -> None:
    """
    Args:
        valloss: {
        '125M-50k': {
            'ln': 3.208740472793579,
            'dyt': {
                0.25: 3.803507089614868,
                0.5: 5.606529712677002,
                1.0: 3.3254964351654053,
                2.0: 3.2982568740844727,
                3.0: 3.279404401779175,
                4.0: 3.2653634548187256,
                5.0: 3.2565524578094482
            },
            [..]
        }
    """
    _, ax = plt.subplots(1, 2, figsize=(2*6,4))

    # limits
    XMIN = -5.5 if xmin is None else xmin 
    XMAX = 8.5 if xmax is None else xmax 
    YMIN = 2.5 if ymin is None else ymin 
    YMAX = 6.5 if ymax is None else ymax

    # shade
    if shade is not None:
        assert isinstance(shade, list) and len(shade) == 2
        for i in range(2):
            assert isinstance(shade[i], list) and len(shade[i]) == 2
            clr = CLR['dyt'] if i == 0 else CLR['dyisrusp']
            ax[i].fill_between(x=shade[i], y1=[YMIN, YMIN], y2=[YMAX, YMAX], color=clr, edgecolor=clr, facecolor=clr, alpha=.1)

    # lines
    for i in range(2):
        ax[i].plot([0, 0], [YMIN, YMAX], linestyle='--', color='gray')
    for i in range(len(ax)):
        if 'ln' in valloss:
            ax[i].plot([XMIN, XMAX], [valloss['ln'], valloss['ln']], linestyle='--', color=CLR['ln'], label=LABEL['ln'])

    # alpha
    if 'dyt' in valloss:
        ax[0].plot(valloss['dyt'].keys(), valloss['dyt'].values(), marker=marker, color=CLR['dyt'], label=LABEL['dyt'])
    xlabel = r'$\alpha_0 ~|~ \widehat \alpha_0$' if additional else r'$\alpha_0$'
    ax[0].set_xlabel(xlabel)
    
    # beta
    if 'dyisrusp' in valloss:
        ax[1].plot(valloss['dyisrusp'].keys(), valloss['dyisrusp'].values(), marker=marker, color=CLR['dyisrusp'], label=LABEL['dyisrusp'])
    xlabel = r'$\beta_0 ~|~ \widehat \beta_0$' if additional else r'$\widehat \beta_0$'
    ax[1].set_xlabel(xlabel)
   
    if additional is True:
        if 'dytsp' in valloss:
            ax[0].plot(valloss['dytsp'].keys(), valloss['dytsp'].values(), marker=marker, color=CLR['dytsp'], label=LABEL['dytsp'])
        if 'dyisru' in valloss:
            ax[1].plot(valloss['dyisru'].keys(), valloss['dyisru'].values(), marker=marker, color=CLR['dyisru'], label=LABEL['dyisru'])

    # diverged
    for variant in ['dyt']:
        if variant in valloss:
            diverged_keys = [key for key in valloss[variant].keys() if valloss[variant][key] is None]
            ax[0].plot(
                diverged_keys, 
                [YMIN + 0.95*(YMAX-YMIN)]*len(diverged_keys), 
                marker=marker, 
                markerfacecolor='white',
                linestyle='',
                color=CLR[variant]
            )
    for variant in ['dyisrusp']:
        if variant in valloss:
            diverged_keys = [key for key in valloss[variant].keys() if valloss[variant][key] is None]
            ax[1].plot(
                diverged_keys, 
                [YMIN + 0.95*(YMAX-YMIN)]*len(diverged_keys), 
                marker=marker, 
                markerfacecolor='white',
                linestyle='',
                color=CLR[variant]
            )
    
    # general
    for i in range(len(ax)):
        ax[i].set_ylabel(r'$\mathcal{L}_{\rm val}$')
        ax[i].legend()
        ax[i].set_ylim([YMIN, YMAX])
        ax[i].set_xlim([XMIN, XMAX])
        if title is not None:
            ax[i].set_title(title)

def _compute_metrics(_df: pd.DataFrame) -> dict[str, float]:
    _metrics = {}

    _metrics['L1'] = float(np.nanmin(np.diag(_df)))
    _metrics['L2'] = float(np.nanmin(_df.min()))
    _metrics['L12'] = _metrics['L1'] - _metrics['L2']

    values_1d = np.diag(_df)
    number_of_all_runs_1d = len(values_1d)
    instable_values_1d = [value for value in values_1d if value > _metrics['L1'] * 1.1]
    stable_values_1d = [value for value in values_1d if value <= _metrics['L1'] * 1.1]
    number_of_instable_runs_1d = len(instable_values_1d)
    _metrics['I1'] = number_of_instable_runs_1d / number_of_all_runs_1d
    _metrics['S1'] = float(np.mean(stable_values_1d) - _metrics['L1'])

    values_2d = _df.values.flatten()
    number_of_all_runs_2d = len(values_2d)
    instable_values_2d = [value for value in values_2d if value > _metrics['L2'] * 1.1]
    stable_values_2d = [value for value in values_2d if value <= _metrics['L2'] * 1.1]
    number_of_instable_runs_2d = len(instable_values_2d)
    _metrics['I2'] = number_of_instable_runs_2d / number_of_all_runs_2d
    _metrics['S2'] = float(np.mean(stable_values_2d) - _metrics['L2'])

    return _metrics

def plot_attn(resultsattn, subset: str = 'all', df_min_col: dict = {}, flip: bool = False, debug: bool = False, diff: bool = False, save_as: str = ''):
    diff_plot = flip is True and diff is True and subset == 'main'
    nr_plots = 3 if diff_plot is True else 2

    if subset == 'all':
        _, ax = plt.subplots(2,nr_plots,figsize=(nr_plots*5,10))
        variants = ['dyt', 'dyisru', 'dytsp', 'dyisrusp']
    else:
        _, ax = plt.subplots(1,nr_plots,figsize=(nr_plots*5,5))
        if subset == 'main':
            variants = ['dyt', 'dyisrusp']
        elif subset == 'alt':
            variants = ['dytsp', 'dyisru']
        else:
            raise ValueError(f'subset = {subset} not defined. needs to be all, main, alt.')

    df_save, metrics = {}, {}
    for idx, variant in enumerate(variants):
        reverse = True if (flip and variant.startswith('dyisru')) else False

        df = pd.DataFrame(resultsattn[variant])
        reordered_idx = sorted(df.index, key=lambda x: float(x), reverse=reverse)
        reordered_col = sorted(df.columns, key=lambda x: float(x), reverse=reverse)
        df = df.reindex(columns=reordered_col, index=reordered_idx)
        if debug is True:
            print(df)

        if diff_plot is True:
            df_save[variant] = df

        i = int(idx > 1)
        j = idx % 2
        _ax = ax[i,j] if subset == 'all' else ax[j]

        if len(df):
            metrics[variant] = _compute_metrics(df)

            vmin = np.nanmin(df.min())
            vmax = vmin + 0.1
            if 0:
                datamax = np.nanmax(df.max())
                print(f'> variant = {variant.ljust(8)}: min = {vmin:.3f}, max = {datamax:.3f} (+{100*(datamax-vmin)/vmin:.2}%)')
            _df_min_col = df.min().to_frame()
            _df_min_col.columns = ['col']
            diag = [df.loc[idx][idx] for idx in _df_min_col.index]
            diag_min = min(diag)
            _df_min_col['diag'] = diag
            _df_min_col.loc['global'] = {'col': vmin, 'diag': diag_min}
            _df_min_col['diff'] = _df_min_col[['col', 'diag']].apply(lambda x: f'{(x[0] - x[1])*100:.2f}%', axis=1)

            cmap = sns.dark_palette(CLR[variant], reverse=True, as_cmap=True)
            sns.heatmap(df, annot=True, fmt='.3g', vmin=vmin, vmax=vmax, ax=_ax, cmap=cmap)
            if subset == 'all':
                if i == 0 and j == 0:
                    xlabel = r'$\alpha_0$'
                    ylabel = r'$\alpha_0^{(\rm attn)}$' 
                elif i == 1 and j == 0:
                    xlabel = r'$\widehat \alpha_0$'
                    ylabel = r'$\widehat \alpha_0^{(\rm attn)}$' 
                elif i == 0 and j == 1:
                    xlabel = r'$\beta_0$'
                    ylabel = r'$\beta_0^{(\rm attn)}$'
                elif i == 1 and j == 1:
                    xlabel = r'$\widehat \beta_0$'
                    ylabel = r'$\widehat \beta_0^{(\rm attn)}$'
            elif subset == 'main':
                if j == 0:
                    xlabel = r'$\alpha_0$'
                    ylabel = r'$\alpha_0^{(\rm attn)}$' 
                elif j == 1:
                    xlabel = r'$\widehat \beta_0$'
                    ylabel = r'$\widehat \beta_0^{(\rm attn)}$'
            elif subset == 'alt':
                if j == 0:
                    xlabel = r'$\widehat \alpha_0$'
                    ylabel = r'$\widehat \alpha_0^{(\rm attn)}$' 
                elif j == 1:
                    xlabel = r'$\beta_0$'
                    ylabel = r'$\beta_0^{(\rm attn)}$'
            _ax.set(xlabel=xlabel, ylabel=ylabel, title=LABEL[variant])
            _ax.invert_yaxis()
            _ax.collections[0].cmap.set_bad('0')
        else:
            _ax.set_xticks([])
            _ax.set_yticks([])

        df_min_col[variant] = _df_min_col

    if diff_plot is True:
        df_save[variants[1]].index = df_save[variants[0]].index
        df_save[variants[1]].columns = df_save[variants[0]].columns
        df_save['diff'] = df_save[variants[1]] - df_save[variants[0]]
        cmap = sns.diverging_palette(120, 240, as_cmap=True)  # 240: blue, 120: green
        sns.heatmap(df_save['diff'], annot=True, fmt='.2f', vmin=-0.1, vmax=0.1, ax=ax[-1], cmap=cmap)
        ax[-1].set(title=f"{LABEL['dyisrusp']} - {LABEL['dyt']}")
        ax[-1].invert_yaxis()

    _save_figure(save_as)

    return df_min_col, metrics

def plot_alpha_beta(alphabeta, title=None, xmin=None, xmax=None, ymin=None, ymax=None, marker='o', additional=False, minmax=False) -> None:
    """
    Args:
        alphabeta: {
        '125M-50k': {
            'ln': None,
            'dyt': {
                0.25: {'mean': 3.803507089614868, 'std': ..},
                0.5: {'mean': 5.606529712677002, 'std': ..},
                1.0: {'mean': 3.3254964351654053, 'std': ..},
                2.0: {'mean': 3.2982568740844727, 'std': ..},
                3.0: {'mean': 3.279404401779175, 'std': ..},
                4.0: {'mean': 3.2653634548187256, 'std': ..},
                5.0: {'mean': 3.2565524578094482, 'std': ..},
            },
            [..]
        }
    """
    _, ax = plt.subplots(1, 2, figsize=(2*6,4))

    # limits
    XMIN = -5.5 if xmin is None else xmin 
    XMAX = 8.5 if xmax is None else xmax 
    YMIN = -10.5 if ymin is None else ymin 
    YMAX = 10.5 if ymax is None else ymax

    # lines
    for i in range(2):
        ax[i].plot([0, 0], [YMIN, YMAX], linestyle='--', color='gray')
        ax[i].plot([YMIN, YMAX], [YMIN, YMAX], linestyle='--', color='gray')

    # alpha
    if 'dyt' in alphabeta:
        if minmax is True:
            ax[0].fill_between(x=alphabeta['dyt']['mean'].keys(), y1=alphabeta['dyt']['min'].values(), y2=alphabeta['dyt']['max'].values(), color=CLR['dyt'], edgecolor=CLR['dyt'], facecolor=CLR['dyt'], alpha=.1)
        ax[0].plot(alphabeta['dyt']['mean'].keys(), alphabeta['dyt']['mean'].values(), marker=marker, color=CLR['dyt'], label=LABEL['dyt'])
    xlabel = r'$\alpha_0 ~|~ \widehat \alpha_0$' if additional else r'$\alpha_0$'
    ylabel = r'$\alpha ~|~ \widehat \alpha$' if additional else r'$\alpha$'
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    
    # beta
    if 'dyisrusp' in alphabeta:
        if minmax is True:
            ax[1].fill_between(x=alphabeta['dyisrusp']['mean'].keys(), y1=alphabeta['dyisrusp']['min'].values(), y2=alphabeta['dyisrusp']['max'].values(), color=CLR['dyisrusp'], edgecolor=CLR['dyisrusp'], facecolor=CLR['dyisrusp'], alpha=.1)
        ax[1].plot(alphabeta['dyisrusp']['mean'].keys(), alphabeta['dyisrusp']['mean'].values(), marker=marker, color=CLR['dyisrusp'], label=LABEL['dyisrusp'])
    xlabel = r'$\beta_0 ~|~ \widehat \beta_0$' if additional else r'$\widehat \beta_0$'
    ylabel = r'$\beta ~|~ \widehat \beta$' if additional else r'$\widehat \beta$'
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
   
    if additional is True:
        if 'dytsp' in alphabeta:
            if minmax is True:
                ax[0].fill_between(x=alphabeta['dytsp']['mean'].keys(), y1=alphabeta['dytsp']['min'].values(), y2=alphabeta['dytsp']['max'].values(), color=CLR['dytsp'], edgecolor=CLR['dytsp'], facecolor=CLR['dytsp'], alpha=.1)
            ax[0].plot(alphabeta['dytsp']['mean'].keys(), alphabeta['dytsp']['mean'].values(), marker=marker, color=CLR['dytsp'], label=LABEL['dytsp'])
        if 'dyisru' in alphabeta:
            if minmax is True:
                ax[1].fill_between(x=alphabeta['dyisru']['mean'].keys(), y1=alphabeta['dyisru']['min'].values(), y2=alphabeta['dyisru']['max'].values(), color=CLR['dyisru'], edgecolor=CLR['dyisru'], facecolor=CLR['dyisru'], alpha=.1)
            ax[1].plot(alphabeta['dyisru']['mean'].keys(), alphabeta['dyisru']['mean'].values(), marker=marker, color=CLR['dyisru'], label=LABEL['dyisru'])
    
    # general
    for i in range(len(ax)):

        ax[i].legend()
        ax[i].set_ylim([YMIN, YMAX])
        ax[i].set_xlim([XMIN, XMAX])
        if title is not None:
            ax[i].set_title(title)


def _get_color(_results, _variant, _key):
    if _results is None:
        clr = CLR[_variant]
    else:
        cmap = sns.dark_palette(CLR[_variant], reverse=True, as_cmap=True)
        vmin = min(_results[_variant].values())
        vmax = vmin + 0.1  
        loss = _results[_variant][_key]
        clr_value = min(255, int((loss - vmin)/(vmax-vmin) * 255))
        clr = cmap(clr_value)
    return clr

def plot_alpha_beta_layers(alphabeta, title=None, xmin=None, xmax=None, ymin=None, ymax=None, marker='o', additional=False, minmax=False, results=None, save_as: str = '') -> None:
    """
    Args:
        alphabeta: {
        '125M-50k': {
            'ln': None,
            'dyt': {
                0.25: {'mean': 3.803507089614868, 'std': ..},
                0.5: {'mean': 5.606529712677002, 'std': ..},
                1.0: {'mean': 3.3254964351654053, 'std': ..},
                2.0: {'mean': 3.2982568740844727, 'std': ..},
                3.0: {'mean': 3.279404401779175, 'std': ..},
                4.0: {'mean': 3.2653634548187256, 'std': ..},
                5.0: {'mean': 3.2565524578094482, 'std': ..},
            },
            [..]
        }
    """
    _, ax = plt.subplots(1, 2, figsize=(2*6,4))

    # limits
    XMIN = -1 if xmin is None else xmin 
    XMAX = 49 if xmax is None else xmax 
    YMIN = -10.5 if ymin is None else ymin 
    YMAX = 10.5 if ymax is None else ymax          

    # alpha
    if 'dyt' in alphabeta:
        keys = list(alphabeta['dyt']['layers'].keys())
        for i, key in enumerate(keys):
            clr = _get_color(results, 'dyt', key)
            x = [int(elem) for elem in alphabeta['dyt']['layers'][key].keys()]
            y = [float(elem) for elem in alphabeta['dyt']['layers'][key].values()]
            if i == 0:
                ax[0].plot(x, y, marker=marker, color=CLR['dyt'], label=LABEL['dyt']) # only for label color
            ax[0].plot(x, y, marker=marker, color=clr)
    ylabel = r'$\alpha ~|~ \widehat \alpha$' if additional else r'$\alpha$'
    ax[0].set_ylabel(ylabel)


    # beta
    if 'dyisrusp' in alphabeta:
        keys = list(alphabeta['dyisrusp']['layers'].keys())
        for i, key in enumerate(keys):
            clr = _get_color(results, 'dyisrusp', key)
            x = [int(elem) for elem in alphabeta['dyisrusp']['layers'][key].keys()]
            y = [float(elem) for elem in alphabeta['dyisrusp']['layers'][key].values()]
            if i == 0:
                ax[1].plot(x, y, marker=marker, color=CLR['dyisrusp'], label=LABEL['dyisrusp']) # only for label color
            ax[1].plot(x, y, marker=marker, color=clr)
    ylabel = r'$\beta ~|~ \widehat \beta$' if additional else r'$\widehat \beta$'
    ax[1].set_ylabel(ylabel)


    if additional is True:
        if 'dytsp' in alphabeta:
            keys = list(alphabeta['dytsp']['layers'].keys())
            for i, key in enumerate(keys):
                x = [int(elem) for elem in alphabeta['dytsp']['layers'][key].keys()]
                y = [float(elem) for elem in alphabeta['dytsp']['layers'][key].values()]
                ax[0].plot(x, y, marker=marker, color=CLR['dytsp'], label=LABEL['dytsp'] if i == 0 else None)
        if 'dyisru' in alphabeta:
            keys = list(alphabeta['dyisru']['layers'].keys())
            for i, key in enumerate(keys):
                x = [int(elem) for elem in alphabeta['dyisru']['layers'][key].keys()]
                y = [float(elem) for elem in alphabeta['dyisru']['layers'][key].values()]
                ax[1].plot(x, y, marker=marker, color=CLR['dyisru'], label=LABEL['dyisru'] if i == 0 else None)
        
    # general
    for i in range(len(ax)):
        xlabel = r'$\widetilde L$'
        ax[i].set_xlabel(xlabel)
        ax[i].legend()
        ax[i].set_ylim([YMIN, YMAX])
        ax[i].set_xlim([XMIN, XMAX])
        if title is not None:
            ax[i].set_title(title)

    _save_figure(save_as)