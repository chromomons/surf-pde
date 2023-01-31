import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()

unif_ref = 2
max_nref = 4

modes = ["old", "new", "total"]

fs = []

for mode in modes:
    f = open(f"{mode}-data.txt", "r")
    fs.append(f)

ts = {}
cg_iters = {}

fig, axs = plt.subplots(5, 1)
fig.set_size_inches(12, 25, forward=True)
plt.subplots_adjust(hspace=0.4)

markers = {"old": "b-", "new": "r-", "total": "k--"}

for nref in range(max_nref+1):
    h = 2 * 2 ** (-unif_ref - nref)
    cg_max = 0
    cur_ax = axs[nref]
    cur_ax.set_title(rf"$h = 2^{{{-nref - 2}}}$")
    for j, mode in enumerate(modes):
        ts = [float(s) for s in fs[j].readline().strip("[]\n").split(",")]
        cgs = [float(s) for s in fs[j].readline().strip("[]\n").split(",")]
        if np.max(cgs) > cg_max:
            cg_max = np.max(cgs)
        cur_ax.plot(range(len(ts)), cgs, markers[mode], label=f"{mode} | {np.mean(cgs):.2E}")
        # cur_ax.plot(ts, cgs, markers[mode], label=f"{mode} | {np.mean(cgs):.2E}")
    cur_ax.set_ylim([1e2, 1e6])
    cur_ax.set_yscale('log')
    if nref == max_nref:
        cur_ax.set_xlabel('time step #')
        # cur_ax.set_xlabel('time')
    if nref == max_nref // 2:
        cur_ax.set_ylabel('cg_iter')
    cur_ax.legend(loc='upper right')
    # plt.savefig(f'cg_iter_test_ref_lvl_{nref}.png')
    # plt.show()

plt.show()

for fl in fs:
    fl.close()
