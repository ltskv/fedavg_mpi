import os
import re
from math import floor, ceil

import matplotlib.pyplot as plt
import numpy as np


HERE = os.path.abspath(os.path.dirname(__file__))
LOGS = os.path.join(HERE, '../../docs/logs/')


datasets = {
    'moby': {
        'idx': 0,
        'name': 'Moby Dick',
        'words': '200k',
        'target': 8.4,
        'lim': (16000, 320000)
    },
    'wiki': {
        'name': 'English Wikipedia',
        'words': '90M',
        'idx': 1,
        'target': 8.3,
        'lim': (16000, 360000)
    }
}


def s(n):
    return 's' if n > 1 else ''


def idx_of(l, cond=lambda x: x):
    try:
        return next(i for i, e in enumerate(l) if cond(e))
    except StopIteration:
        return -1


def meta_from_fn(fn):
    m = re.search(r'(.+)_(\d+)_learner_(\d+)_pp', fn)
    return (lambda x: (x[0], int(x[1]), int(x[2])))(
        m.group(1,2,3)
    )



if __name__ == '__main__':
    files = sorted(os.listdir(LOGS), key= lambda x: meta_from_fn(x)[1])

    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, wspace=0.18)
    axs = fig.subplots(1, len(datasets))
    pp_speedup = {
        'wiki': [],
    }
    l_speedup = {
        'moby': [],
        'wiki': [],
    }

    for fn in files:
        name, learners, pipelines = meta_from_fn(fn)
        if learners == 16:
            continue
        with open(os.path.join(LOGS, fn)) as f:
            lines = f.readlines()
        matches = [re.search(r'windows (\d+) validation loss (\d+\.\d+)', l)
                   for l in lines]
        matches = [m for m in matches if m is not None]
        win_loss = [
            (lambda x: (int(x[0]), float(x[1])))(m.group(1, 2)) for m in matches
        ]
        windows, loss = zip(*win_loss)
        axs[datasets[name]['idx']].plot(
            windows[1:], loss[1:], linestyle='-' * (1 + (pipelines>1)),
            color=f'C{learners // 2}',
            label=f'{learners} Learner{s(learners)},'
            f' {pipelines} Pipeline{s(pipelines)}'
        )
        ttt = windows[idx_of(loss, lambda l: l < datasets[name]['target'])]
        if (pipelines > 1 or learners == 1) and name == 'wiki':
            pp_speedup[name].append((pipelines, ttt))
        if pipelines == 1:
            l_speedup[name].append((learners, ttt))

    for d in datasets.values():
        a = axs[d['idx']]
        a.set_xlabel('Context Windows per Learner')
        a.set_ylabel('Validation Loss')
        a.set_xticks([windows[1]] + [*range(0, 300001, 100000)])
        a.set_xlim(*d['lim'])
        a.set_title(f'{d["name"]} (~{d["words"]} words)')
        a.legend()
        a.axhline(d['target'], color='k', linestyle=':')

    fig.savefig(os.path.join(HERE, 'fig/datasets.pdf'))

    def speedup_plot(zipped):
        min_f = []
        max_f = []
        min_s = []
        max_s = []
        for z in sorted(zipped, key=lambda x: datasets[x]['idx']):
            d = datasets[z]
            factors, time = zip(*sorted(zipped[z]))
            time = np.asarray(time)
            speedup = time[0] / time
            print(factors, time)
            plt.plot(
                factors, speedup,
                label=f'{d["name"]}, target: {d["target"]}',
                color=f'C{d["idx"]}'
            )
            min_s.append(min(speedup))
            max_s.append(max(speedup))
            min_f.append(min(factors))
            max_f.append(max(factors))

        plt.xlim(min(min_f), max(max_f))
        plt.ylim(min(min_s), max(max_s))
        plt.xticks([*range(min(min_f), max(max_f) + 1)])
        plt.yticks([*range(floor(min(min_s)), ceil(max(max_s)) + 1)])
        plt.legend(loc='upper left')
        plt.grid()

    fig = plt.figure(figsize=(10, 4))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.91, wspace=0.18)

    plt.subplot(121)
    speedup_plot(l_speedup)
    plt.title('Single Pipeline')
    plt.xlabel('Number of Learners')
    plt.ylabel(f'Speedup to Target')

    plt.subplot(122)
    speedup_plot(pp_speedup)
    plt.title('Multiple Pipelines')
    plt.xlabel('Number of Pipelines')
    plt.ylabel(f'Speedup to Target')

    plt.savefig(os.path.join(HERE, 'fig/speedups.pdf'))
    plt.show()
