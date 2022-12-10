#!/usr/bin/python3

# Desc: Plots a single benchmark
# Usage: ./single_plot.py <CSV File> <Output File>

import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

if len(sys.argv) != 3:
    print("Single Benchmark Plotting Tool")
    print("Arguments:")
    print("  Input CSV: The CSV file to plot")
    print("  Output File: The filepath of the generated plot")
    exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def add_median_labels(ax, precision='.2f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[4:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] -
                      median.get_xdata()[0]) == 0 else y
        ax.text(x,
                y * 1.1,
                f'{value:{precision}}',
                ha='center',
                va='bottom',
                fontweight='bold',
                color='black')


dt = pd.read_csv(input_file)
sns.set(style="darkgrid")
sns.set(font_scale=1.25)

plt.figure(figsize=(8, 4))
box_plot = sns.boxplot(data=dt, notch=True)

ax = box_plot.axes
ax.set(ylabel='Runtime [ms]')

add_median_labels(ax)

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
