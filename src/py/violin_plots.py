#!/usr/bin/env python3

import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def dataframes(targets, hic, targets_hic):
	targets = pd.read_csv(targets, sep='\t')
	hic = pd.read_csv(hic, sep='\t')
	targets_hic = pd.read_csv(targets_hic, sep='\t')

	targets_auroc, hic_auroc, targets_hic_auroc = list(targets.loc[:, 'auroc']), list(hic.loc[:, 'auroc']), list(targets_hic.loc[:, 'auroc'])

	auroc = targets_auroc + hic_auroc + targets_hic_auroc

	labels = ['targets', 'hic', 'targets_hic']

	l_labels =[]
	for l in labels:
		for i in range(164):
			l_labels.append(l)


	concat_data = pd.DataFrame({'auroc': auroc, 'strategy': l_labels})
	
	return concat_data


def violin(df, output):
	sns.set(style = 'whitegrid')
	sns.violinplot(x = 'strategy',
		y = 'auroc',
		data = df)
	plt.savefig(output)


if __name__ == '__main__':
	# set paths to auroc metrics for each strategy
	targets = sys.argv[1]
	hic = sys.argv[2]
	targets_hic = sys.argv[3]
	output = sys.argv[4]

	df = dataframes(targets, hic, targets_hic)
	violin(df, output)
