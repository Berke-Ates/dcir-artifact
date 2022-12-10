#!/usr/bin/python3

# Desc: Plots multiple benchmarks
# Usage: ./multi_plot.py [<CSV File> <CSV File> ...] <Output File>

import sys
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from math import ceil

if len(sys.argv) < 3:
    print("Multi Benchmark Plotting Tool")
    print("Arguments:")
    print("  Input CSVs: A list of CSV files to plot (at least 4)")
    print("  Output File: The filepath of the generated plot")
    exit(1)

input_files = sys.argv[1:-1]
output_file = sys.argv[-1]

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

dt = []
num_bench = len(input_files)

if (num_bench < 4):
    print("This plotter requires at least 4 CSV files")
    exit(1)

rows = 3
cols = ceil(num_bench / rows)
color = ['#4D678D', '#4D678D', '#4D678D', '#4D678D', '#EA7878']

fig, ax = plt.subplots(rows, cols, sharex='row')
fig.set_figheight(7)
fig.set_figwidth(17)
fig.supylabel('Runtime [s]')

sns.set(style="darkgrid")

for i in range(num_bench):
    dt.append(pd.read_csv(input_files[i]))

for i in range(rows):
    for j in range(cols):
        if (i * cols + j >= num_bench):
            ax[i, j].axis('off')
            continue

        sns.barplot(data=dt[i * cols + j],
                    palette=color,
                    estimator=np.median,
                    ax=ax[i, j])

        bench_name = os.path.splitext(
            os.path.basename(input_files[i * cols + j]))[0]

        ax[i, j].set_title(bench_name)
        ax[i, j].set_xticklabels(ax[i, j].get_xticklabels(),
                                 rotation=90,
                                 ha="center")

        if (i < rows - 1):
            ax[i, j].set(xticklabels=[])

plt.tight_layout()
plt.savefig(output_file, dpi=300, bbox_inches='tight')
