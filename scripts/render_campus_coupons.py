"""Plot the completed campus simulation; never generate or re-fit outcomes."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/candidate-ranking-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'reports/campus-coupons-v1')
    args = parser.parse_args()
    data = json.loads((args.report / 'results.json').read_text())
    if not data['synthetic'] or not data['integrity']['passed']:
        raise SystemExit('Only verified, explicitly synthetic results may be plotted')
    rows = data['experiments']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(11, 5.4))
    fig.subplots_adjust(left=.30, right=.95, top=.70, bottom=.28)
    labels = ['A/A control\nBoth policies target clicks',
              'A/B: campus time constraint\nClick target vs. coupon target',
              'A/B: travel constraint removed\nSame frozen policies; no retraining']
    for i, row in enumerate(rows):
        for metric, offset, color, marker in [('click', .12, '#747a80', 's'),
                                               ('coupon', -.12, '#176b68', 'o')]:
            effect = row['effects'][metric]
            mid = effect['difference_pp']
            low, high = effect['conditional_ci95_pp']
            y = 2-i+offset
            ax.errorbar(mid, y, xerr=[[mid-low], [high-mid]], fmt=marker,
                        color=color, capsize=3, markersize=5, elinewidth=1.3,
                        label=('Click' if metric == 'click' else 'Coupon redeemed') if i == 0 else None)
            ax.annotate(f'{mid:+.2f}', (high, y), xytext=(7, 0),
                        textcoords='offset points', va='center', color=color, fontsize=10)
    ax.axvline(0, color='#999', linestyle='--', linewidth=.8, zorder=0)
    ax.set_yticks([2, 1, 0], labels)
    ax.tick_params(axis='y', length=0, pad=12)
    ax.set_xlim(-21, 21)
    ax.set_ylim(-.5, 2.5)
    ax.set_xlabel('B − A per scheduled visit (percentage points)', labelpad=9)
    ax.legend(loc='lower left', bbox_to_anchor=(-.01, 1.04), ncol=2, frameon=False)
    fig.text(.035, .94, 'The preferred ranking reverses with the travel assumption', fontsize=16)
    fig.text(.035, .88, 'Campus coupon simulation · 3,000 synthetic users × 8 visits per condition · one sponsored slot',
             fontsize=10, color='#4c535b')
    fig.text(.035, .08, 'Points: differences in user means. Bars: 95% user-bootstrap intervals, conditional on the invented simulator.\n'
             'The same cohort is reused across conditions. No human participants, real purchases or LLM respondent calls.',
             fontsize=9, color='#4c535b', linespacing=1.6)
    for ext in ['png', 'svg']:
        fig.savefig(args.report / f'campus-coupons.{ext}', dpi=200, facecolor='white')
    plt.close(fig)
    print(args.report / 'campus-coupons.png')


if __name__ == '__main__':
    main()
