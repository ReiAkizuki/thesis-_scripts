import random
import math
import matplotlib.pyplot as plot
from sklearn.cluster import KMeans

plot.rcParams["figure.figsize"] = [8, 8]
plot.rcParams['font.size'] = 6

data = []
# (x-2)^2 + (y-2)^2 <= 1
for i in range(0,1000):
	x = random.uniform(-1, 1)
	y = math.sqrt(1 - x**2) * random.uniform(-1, 1) + 2
	data.append([x + 2, y])

# (x-3)^2 + (y-4)^2 <= 1
for i in range(0,1000):
	x = random.uniform(-1, 1)
	y = math.sqrt(1 - x**2) * random.uniform(-1, 1) + 5
	data.append([x + 3, y])

# (x-4)^2 + (y-3)^2 <= 1
for i in range(0,1000):
	x = random.uniform(-1, 1)
	y = math.sqrt(1 - x**2) * random.uniform(-1, 1) + 3
	data.append([x + 5, y])

for i in range(0, 3000):
	x = data[i][0]
	y = data[i][1]
	plot.plot(x, y, ms=5.0, marker='o', color='0.0')

plot.savefig('test_sse_plot_data.png')

# Clear the current figure.
plot.clf()

distortions = []

# 1~10クラスタまでのSSE値を求めて図示（エルボー図）
for i  in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    km.fit(data)
    distortions.append(km.inertia_)

plot.plot(range(1,11), distortions, marker='o', color='0.0')
plot.xlabel('クラスタ数', size=14)
plot.ylabel('クラスタ内誤差平方和（SSE）', size=14)
plot.savefig('test_sse_plot.png')

# Clear the current figure.
plot.clf()
