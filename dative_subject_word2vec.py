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
from sklearn.cluster import KMeans

plot.rcParams["figure.figsize"] = [8, 8]
plot.rcParams['font.size'] = 6

distortions = []

# 1~10クラスタまでのSSE値を求めて図示（エルボー図）
for i  in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(data)
    distortions.append(km.inertia_)

plot.plot(range(1,11), distortions, marker='o', color='0.0')
plot.xlabel('クラスタ数', size=14)
plot.ylabel('クラスタ内誤差平方和（SSE）', size=14)
plot.savefig('sse_plot.png')

# Clear the current figure.
plot.clf()

# k-meansモデルを用いてクラスタリング、クラスター数はエルボー図から3に決定した
model = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
model.fit(data)
# 変数名被り防止
kluster=model.predict(data)

pca = PCA(n_components=2)
pca.fit(data)
data_pca = pca.transform(data)

i = 0
word_list = []
# python3.7以降dictは順序を保持する
for word, val in word_dic.items():
	if val['skip']:
		continue

	word_list.append([word] + list(val.values()) + [data_pca[i]] + [kluster[i]])
	i += 1

word_df = pd.DataFrame(word_list, columns=['word', 'num', 'dsc_num', 'skip', 'pca_vec', 'cluster'])
print(word_df)
word_df.to_csv('word_df.csv', sep=',')

markers = ['.', '+', 'x']
i = 0
for key, val in word_dic.items():
	if val['skip']:
		continue

	x = data_pca[i][0]
	y = data_pca[i][1]
	# x = 0で0.9, x = 1で0.09, x = 0.5で0.2
	color = 1 / (1.1 + 10 * val['dsc_num'] / val['num'])
	plot.plot(x, y, ms=5.0, zorder=2, marker=markers[kluster[i]], color=str(color))
	if val['dsc_num'] > 0:
		plot.annotate(key, (x, y), size=14)
		print(key)
		print(val['dsc_num'] / val['num'])

	i += 1

plot.savefig('word-map.png')

plot.clf()

# クラスタごとに色分け
i = 0
for key, val in word_dic.items():
	if val['skip']:
		continue

	x = data_pca[i][0]
	y = data_pca[i][1]
	color = 1 - 1 / (kluster[i] + 1)
	plot.plot(x, y, ms=5.0, zorder=2, marker=markers[kluster[i]], color=str(color))
	if val['dsc_num'] > 0:
		plot.annotate(key, (x, y), size=14)

	i += 1

plot.savefig('word-map2.png')

