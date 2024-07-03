import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE
import torch

g = torch.load('/root/autodl-tmp/FairLLM4Graph/data/citeseer/g.pt')
path = '/root/autodl-tmp/FairLLM4Graph/checkpoints/citeseer/bert-base-uncased/text_embeddings_True.pt'

embeds = torch.load(path)
origin = g.x
label = g.y

def plot_embedding(data, label):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	# x_min, x_max = np.min(data, 0), np.max(data, 0)
	# data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
	# data = torch.from_numpy(data)
	fig = plt.figure()		# 创建图形实例
	ax = plt.subplot(111)		# 创建子图
	label_mx = max(label)
	for i in range(label_mx + 1):
		idx = torch.nonzero(label == i, as_tuple=False).squeeze(dim=-1)
		sub_data = data[idx]
		plt.scatter(sub_data[:, 0], sub_data[:, 1], c=plt.cm.Set1(i / 8))
	plt.xticks()		# 指定坐标的刻度
	plt.yticks()
	plt.title('t-SNE', fontsize=14)
	# 返回值
	return fig

ts = TSNE(n_components=2, init='pca', random_state=0)

res1, res2 = ts.fit_transform(embeds), ts.fit_transform(origin)
fig1 = plot_embedding(res1, label)
plt.savefig('bert.jpg')

# fig2 = plot_embedding(res2, label)
# plt.savefig('origin.jpg')