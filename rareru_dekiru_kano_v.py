import pandas as pd
import numpy as np
import os
import glob
import re

base_path = os.path.dirname(os.path.abspath(__file__))
# コーパスの格納場所によって変更する
corpus_base = 'Documents/livedoor_news_corpus/text'
os.chdir(os.environ['HOME'])
os.chdir(corpus_base)

article_list = []

#フォルダ内のテキストファイルを全てサーチ

for p in glob.glob('**/*.txt'):
	#第二階層フォルダ名がニュースサイトの名前になっているので、それを取得
	media = str(p).split('/')[0]
	file_name = str(p).split('/')[1]

	if not file_name in ['CHANGES.txt', 'LICENSE.txt', 'README.txt']:
		#テキストファイルを読み込む
		with open(p, 'r') as f:
			#テキストファイルの中身を一行ずつ読み込み、リスト形式で格納
			article = f.readlines()
			#不要な改行等を置換処理
			article = [re.sub(r'[\n \u3000]', '', i) for i in article]
			#ニュースサイト名・記事URL・日付・記事タイトル・本文の並びでリスト化
			article_list.append([media, article[0], article[1], article[2], ''.join(article[3:])])

	else:
		continue

os.chdir(base_path)
article_df = pd.DataFrame(article_list)

article_df.to_csv('test_dative_subject.csv', sep=',')

import MeCab
import subprocess

cmd='echo `mecab-config --dicdir`"/mecab-ipadic-neologd"'
path = (subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]).decode('utf-8')
m=MeCab.Tagger('-d {0}'.format(path))

mapping = {
	'える': 'う',
	'ける': 'く',
	'げる': 'ぐ',
	'せる': 'す',
	'ぜる': 'ず',
	'てる': 'つ',
	'でる': 'づ',
	'ねる': 'ぬ',
	'へる': 'ふ',
	'べる': 'ぶ',
	'ぺる': 'ぷ',
	'める': 'む',
	'れる': 'る'
}

# -eru -> -uとして可能動詞にならない動詞
not_kano_list = [ '流れる', 'かける', '入れる', '向ける', '助ける', '合わせる', '届ける', '支える',
	'埋もれる', '揺れる', '伝える', '触れる', '咲かせる', '任せる', '優れる', '込める', 'あふれる',
	'告げる', 'のせる', '隠れる', 'ぶれる', '求める', '捧げる', '笑わせる', '傾げる', '震える', '構える',
	'染める', '恐れる', '育てる', '見上げる', '建てる', '止める', '続ける', '充てる', '分かれる', '整える'
	,'温める', '取り入れる', 'あわせる', '分ける', '知らせる', '辞める', '間違える', 'まぎれる', '紛れる',
	'組み合わせる', '潰れる', 'つぶれる', '与える', 'あたえる', '傾ける', '唱える', 'とげる', '取り付ける',
	'付ける', 'こぎつける', '改める', 'なめる', '問い合わせる', '言い聞かせる', '食わせる', '賑わせる',
	'光らせる', 'みせる', '透ける', 'まかせる', '溶ける', '欠ける', '詰める', 'しかれる', '結びつける',
	'痛める', '寝かせる', '捧げる', '踏み入れる', '伏せる', '引き立てる', 'つける' ]

# 出現した全動詞のリスト
verb_list = []
# 可能/自発のdative subject constructionに出現するVのリスト
candidate_s_verb_list = []
candidate_s_list = []
for text in article_df[4]:
	sentences = re.split('[。！？!?]', text)
	for s in sentences:
		# print(s)
		v_i = []
		word_list = m.parse(s).split('\n')
		morphemes = [''] * (len(word_list) - 2)
		# 最後の2要素は'EOS'と''なので捨てる
		for i, w in enumerate(word_list[:-2]):
			tag = re.split('[,\t]', w)
			if tag[2] == '格助詞' and tag[7] == 'に':
				# Ni
				morphemes[i] = 'n' if re.search('[dkv]', morphemes[i-1]) else morphemes[i-1] + 'n'
			# 〜にはを排除
			elif tag[2] == '係助詞':
				if len(morphemes[i-1]) > 0 and morphemes[i-1][-1] == 'n':
					morphemes[i] = morphemes[i-1][:-1]
				elif re.search('[dkv]', morphemes[i-1]):
					morphemes[i] = ''
				else:
					morphemes[i] = morphemes[i-1]
			elif tag[1] == '動詞' and tag[2] == '自立':
				verb_list.append(tag[7])
				# 出現した動詞の位置を記録する。補助動詞は無視する。
				v_i.append(i)
				if tag[7] == 'できる':
					# Dekiru
					morphemes[i] = 'd' if re.search('[dkv]', morphemes[i-1]) else morphemes[i-1] + 'd'
				elif tag[5] == '一段':
					v = tag[7]
					if len(v) > 2 and v[-2:] in mapping and not v in not_kano_list:
						# 身に着ける, 手に入れるを排除
						if v == '着ける' and word_list[i-2] == '身' and word_list[i-1] == 'に':
							morphemes[i] = 'v' if re.search('[dkv]', morphemes[i-1]) else morphemes[i-1] + 'v'
						if v == '入れる' and word_list[i-2] == '手' and word_list[i-1] == 'に':
							morphemes[i] = 'v' if re.search('[dkv]', morphemes[i-1]) else morphemes[i-1] + 'v'
						else:
							org_v = v[0:-2] + mapping[v[-2:]]
							if org_v == re.split('[,\t]', m.parse(org_v))[7]:
								# Kanou_doushi
								morphemes[i] = 'k' if re.search('[dkv]', morphemes[i-1]) else morphemes[i-1] + 'k'
				else:
					morphemes[i] = 'v' if re.search('[dkv]', morphemes[i-1]) else morphemes[i-1] + 'v'
			elif tag[1] == '動詞' and tag[7] in ['れる', 'られる']:
				if i > 0:
					pred = re.split('[,\t]', word_list[i-1])
					if not re.search('(五段|サ変)', pred[5]):
						morphemes[i] = morphemes[i-1] + 'r'
			elif tag[1] == '動詞' and tag[2] == '非自立':
				# 補助動詞に「られる/できる」がつく形を排除しないようにする
				if len(morphemes[i-1]) > 0 and morphemes[i-1][-1] == 'v':
					morphemes[i] = morphemes[i-1]
				else:
					morphemes[i] = ''
			# 「」、！？などではリセット
			elif tag[1] == '記号':
				morphemes[i] = ''
			else:
				morphemes[i] = '' if re.search('[dkv]', morphemes[i-1]) else morphemes[i-1]
		for j, morpheme in enumerate(morphemes):
			if 'nd' in morpheme:
				# print(s)
				v = re.split('[,\t]', word_list[j])[7]				
				candidate_s_verb_list.append(v)
				candidate_s_list.append([s, v])
			elif 'nk' in morpheme:
				# print(s)
				v = re.split('[,\t]', word_list[j])[7]
				org_v = v[0:-2] + mapping[v[-2:]]
				candidate_s_verb_list.append(org_v)
				candidate_s_list.append([s, org_v])
			elif 'nvr' in morpheme:
				# print(s)
				tmp = j
				while not tmp in v_i:
					tmp -= 1
				v = re.split('[,\t]', word_list[tmp])[7]
				candidate_s_verb_list.append(v)
				candidate_s_list.append([s, v])

candidate_s_verb_df = pd.DataFrame(candidate_s_verb_list)
candidate_s_df = pd.DataFrame(candidate_s_list)
candidate_s_verb_df.to_csv('candidate_s_verbs.csv', sep=',')
candidate_s_df.to_csv('candidate_s.csv', sep=',')
