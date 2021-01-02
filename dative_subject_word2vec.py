# MacOS Catalina 10.15.7 Python 3.8.5
import os
import gensim
import pandas as pd
from gensim.models import KeyedVectors

texts_df = pd.read_csv('candidate_yn.csv')
word_dic = {}
data = []
model = KeyedVectors.load_word2vec_format('/usr/local/lib/entity_vector/entity_vector.model.bin', binary=True)

# dicの作成
for index, text in texts_df.iterrows():
	if len(text) > 3:
		print(text[2])
		word = text[2]
		if word in word_dic:
			word_dic[word]['num'] += 1
		else:
			word_dic[word] = { 'num': 1, 'dsc_num': 0, 'skip': False }
		if text[3] == 'y':
			word_dic[word]['dsc_num'] += 1

for word in word_dic:
	if word in model:
		data.append(model[word])
	else:
		word_dic[word]['skip'] = True

import matplotlib
import matplotlib.pyplot as plot
from sklearn.decomposition import PCA
import json

plot.rcParams["figure.figsize"] = [8, 8]
plot.rcParams['font.size'] = 6

pca = PCA(n_components=2)
pca.fit(data)
data_pca = pca.transform(data)


length = len(data_pca)
i = 0
for key, val in word_dic.items():
	if val['skip']:
		continue

	x = data_pca[i][0]
	y = data_pca[i][1]
	# x = 0で0.9, x = 1で0.09, x = 0.5で0.2
	color = 1 / (1.1 + 10 * val['dsc_num'] / val['num'])
	plot.plot(x, y, ms=5.0, zorder=2, marker=".", color=str(color))
	if val['dsc_num'] > 0:
		plot.annotate(key, (x, y), size=14)
		print(key)
		print(val['dsc_num'] / val['num'])

	i += 1

plot.savefig('word-map.png')
