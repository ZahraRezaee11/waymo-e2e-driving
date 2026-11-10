"""Bar chart of RFS per method -> assets/rfs_per_method.png (numbers from the README table)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BLUE, GREY = '#2a78d6', '#8a8984'
INK, INK2, SURFACE = '#0b0b0b', '#52514e', '#fcfcfb'

# (label, RFS, colour, +- error or None)
test = [
    ('EVQ-L1d + VLM turn routing (mine)',    7.511, BLUE,  None),
    ('EVQ-L1d (mine)',                       7.375, BLUE,  None),
    ('NaiveEMMA baseline',                   7.528, GREY,  None),
    ('AutoVLA paper',                        7.556, GREY,  None),
    ('Poutine',                              7.986, GREY,  None),
]
val = [
    ('EVQ-L1d + VLM turn routing',           7.776, BLUE,  None),
    ('EVQ-L1d',                              7.681, BLUE,  None),
    ('AutoVLA + GRPO, GBM selector of 8',    7.42,  BLUE,  0.09),
    ('AutoVLA + GRPO, greedy',               7.149, BLUE,  None),
    ('Constant velocity',                    7.027, GREY,  None),
    ('Logged trajectory (ceiling)',          8.175, GREY,  None),
]

fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), dpi=200, sharex=True,
                         gridspec_kw={'height_ratios': [len(test), len(val)], 'hspace': 0.32})
fig.patch.set_facecolor(SURFACE)
for ax, rows, title in [(axes[0], test, 'Official test set (leaderboard)'),
                        (axes[1], val, 'Validation set (478 rated frames, local RFS)')]:
    ax.set_facecolor(SURFACE)
    y = list(range(len(rows)))[::-1]
    for yi, (label, v, c, err) in zip(y, rows):
        best = rows is test and yi == y[0]
        ax.barh(yi, v, height=0.62, color=c, edgecolor=SURFACE, linewidth=2)
        if err:
            ax.errorbar(v, yi, xerr=err, color=INK, capsize=3, linewidth=1.2)
        txt = f'{v:.3f}' if err is None else f'{v:.2f} \u00b1 {err:.2f}'
        ax.text(v + (err or 0) + 0.08, yi, txt, va='center', ha='left', fontsize=9, color=INK,
                fontweight='bold' if best else 'normal')
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=9, color=INK)
    ax.set_xlim(0, 10)
    ax.tick_params(axis='x', labelsize=8, colors=INK2); ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, color='#e6e5e0', linewidth=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.spines['bottom'].set_color('#c9c8c2')
    ax.set_title(title, fontsize=9.5, color=INK2, loc='left', pad=6)
axes[1].set_xlabel('Rater Feedback Score (0 to 10, higher is better)', fontsize=9, color=INK2)
fig.suptitle('WOD-E2E: RFS per method, one RTX 5090', fontsize=11, color=INK, x=0.125, ha='left', y=0.97)
axes[1].legend(handles=[Patch(color=BLUE, label='My methods'),
                        Patch(color=GREY, label='Baselines, references and ceiling')],
               loc='upper center', bbox_to_anchor=(0.4, -0.22), ncol=2, frameon=False, fontsize=8)
fig.savefig('assets/rfs_per_method.png', facecolor=SURFACE, bbox_inches='tight')
print('saved assets/rfs_per_method.png')
