"""Standalone research-style figure from audited synthetic A/B aggregates."""
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
    parser.add_argument('--report', type=Path, default=Path(__file__).resolve().parents[1]/'reports/synthetic-ab-v1')
    args = parser.parse_args()
    data = json.loads((args.report/'results.json').read_text())
    if not data['integrity']['passed']:
        raise SystemExit('Do not plot an effect from incomplete or invalid logs')
    rows = data['experiments']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.subplots_adjust(left=.32, right=.95, top=.70, bottom=.32)
    labels = {'aa-check': 'A/A control\nPopularity vs. popularity',
              'ab-main': 'A/B: preferences\nPopularity vs. contextual LightGBM',
              'ab-null': 'A/B: item-independent response\nPopularity vs. contextual LightGBM'}
    for y, row in enumerate(rows):
        center = row['difference_pp']
        low, high = row['conditional_simulation_ci95_pp']
        color = '#176b68' if row['experiment_id'] == 'ab-main' else '#575b60'
        ax.errorbar(center, 2-y, xerr=[[center-low], [high-center]], fmt='o',
                    color=color, capsize=4, elinewidth=1.5, markersize=6)
        ax.annotate(f'{center:+.2f} pp', (high, 2-y), xytext=(10, 0),
                    textcoords='offset points', va='center', fontsize=10, color=color)
    ax.set_yticks([2, 1, 0], [labels[r['experiment_id']] for r in rows])
    ax.tick_params(axis='y', length=0, pad=12)
    ax.axvline(0, color='#8d9298', linestyle='--', linewidth=1, zorder=0)
    ax.set_ylim(-.5, 2.5)
    lo = min(r['conditional_simulation_ci95_pp'][0] for r in rows)
    hi = max(r['conditional_simulation_ci95_pp'][1] for r in rows)
    ax.set_xlim(min(-1.5, lo-1), hi+4)
    ax.set_xlabel('B − A: simulated visit click rate (percentage points)', labelpad=10)
    fig.text(.035, .93, 'What changes when the response assumptions change?', fontsize=16)
    fig.text(.035, .85, '3,000 synthetic users × 8 visits per condition · actual HTTP serving and durable event logs',
             fontsize=10, color='#4c535b')
    fig.text(.035, .07, 'Points: user-level mean difference. Bars: 95% bootstrap intervals conditional on the invented simulator.\n'
             'No human participants, real clicks or LLM respondent calls. These are newly trained synthetic-world policies.',
             fontsize=9, color='#4c535b', linespacing=1.5)
    fig.savefig(args.report/'synthetic-ab.png', dpi=200, facecolor='white')
    fig.savefig(args.report/'synthetic-ab.svg', facecolor='white')
    plt.close(fig)
    print(args.report/'synthetic-ab.png')


if __name__ == '__main__':
    main()
