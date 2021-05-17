# -*- coding: utf-8 -*-
# @Author: 86498
# @Date:   2021-04-25 12:07:18
# @Last Modified by:   86498
# @Last Modified time: 2021-05-17 16:18:34
import numpy as np
import matplotlib.pyplot as plt
class mulCOG(object):
	"""docstring for mulCOG"""
	#初始化函数
	def __init__(self):
		super(mulCOG, self).__init__()
	#输入函数，一些参数如聚类数量k，原始数据dataSet均由此输入，-1代表不顺利，1代表顺利
	def inputFunc(self):
		flag = input("输入0进入案例计算，1进入自定义计算：")
			#案例，为10个供应地挑选2个物流中心
		if flag == '0':
			self.k = 2
			self.num = 10
			self.dataSet = np.loadtxt('多重心法例.txt',delimiter='\t')
			return 1
		elif flag == '1':
			self.k = int(input("请输入要规划的设施数量："))
			self.num = int(input("请输入供应地（销售地）的数量："))
			if self.k >= self.num:
				print("这不合理。计算结束。")
				return -1
			else:
				self.dataSet = np.zeros((self.num,5))
				for i in range(self.num):
					temp = eval(input("请依次输入第{}个供应地（销售地）的x坐标、y坐标、年运输量q、费用比r，用逗号隔开：".format(i+1)))
					self.dataSet[i,0] = i+1
					self.dataSet[i][1:5] = np.array(temp)
				return 1
	#计算欧式距离函数
	def distCalc(self,x,y):
		return np.sqrt(np.sum((x-y)**2))
	#为给定数据集构建一个包含K个随机质心的集合函数
	def randCent(self):
		m,n = self.dataSet.shape
		self.centroids = np.zeros((self.k+1,n))
		# for i in range(self.k):
		# 	index = int(np.random.uniform(0,m)) #
		# 	self.centroids[i,:] = np.array(self.dataSet.loc[index,:])
		s = set()
		for i in range(1,self.k+1):
			while True:
				index = int(np.random.uniform(0,m))
				if index not in s:
					s.add(index)
					break
			self.centroids[i,:] = self.dataSet[index,:]
	#k均值聚类函数，以Z = rdj 做了一些调整
	def justKMeans(self):
		m = np.shape(self.dataSet)[0]  #行的数目
		# 第一列存样本属于哪一簇，第二列存样本的到簇的中心点的误差
		self.clusterAssment = np.mat(np.zeros((m,2)))
		for i in range(m):
			self.clusterAssment[i,0] = -1
		clusterChange = True
		# 第1步 初始化centroids
		self.randCent()
		while clusterChange:
			clusterChange = False
			# 遍历所有的样本（行数）
			for i in range(m):
				minWeigh = 10000000000000000.0
				minIndex = -1
				#第2步 遍历所有的质心，找出最近的质心
				for j in range(1,self.k+1):
				# 计算该样本到质心的权重距离
					#Weigh = self.distCalc(self.centroids[1:3],self.dataSet[i][1:3])*self.dataSet[i][3]*self.dataSet[i][4]
					dist1 = self.distCalc(self.centroids[j][1:3],self.dataSet[i][1:3])
					# print(dist1)
					# print("hi")
					Weigh = dist1*self.dataSet[i][3]*self.dataSet[i][4]
					# print(Weigh)
					# print("bye")
					if Weigh < minWeigh:
						minWeigh = Weigh
						minIndex = j
				# 第 3 步：更新每一行样本所属的簇
				if self.clusterAssment[i,0] != minIndex:
					clusterChange = True
					self.clusterAssment[i,:] = minIndex,minWeigh**2
				else:
					self.clusterAssment[i,1] = minWeigh
			#第 4 步：更新质心
			for j in range(1,self.k+1):
				pointsInCluster = self.dataSet[np.nonzero(self.clusterAssment[:,0].A == j)[0]]  # 获取簇类所有的点
				if len(pointsInCluster) != 0:
					self.centroids[j][1:3] = np.mean(pointsInCluster,axis=0)[1:3]   # 对矩阵的行求均值
		print("k-均值法聚类完毕。")
	#画聚类示意图函数
	def showCluster(self):
			m= self.dataSet.shape[0]
			mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
			if self.k > len(mark):
				print("k值太大了")
				return -1
			# 绘制所有的样本
			for i in range(m):
				markIndex = int(self.clusterAssment[i,0])
				plt.plot(self.dataSet[i,1],self.dataSet[i,2],mark[markIndex])
			mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
			# 绘制质心
			for i in range(self.k):
				plt.plot(self.solu[i,0],self.solu[i,1],mark[i])
			plt.show()
	#迭代求解函数
	def optSolu(self):
		self.solu = np.zeros((self.k,2))
		self.totaCost = []
		self.pIC = []
		for i in range(self.k):
			self.pIC.append(self.dataSet[np.nonzero(self.clusterAssment[:,0].A == i+1)[0]])
		for i in range((self.k)):
			pICinfo = np.array(self.pIC[i])
			weighCol = np.array(pICinfo[:,3])*np.array(pICinfo[:,4])
			weighColX = (np.array(pICinfo[:,1])*weighCol).sum()
			weighColY = (np.array(pICinfo[:,2])*weighCol).sum()
			x0,y0 = weighColX/weighCol.sum(),weighColY/weighCol.sum()
			# dist = np.sqrt((np.array(pICinfo[i][1])-x0)**2+(np.array(pICinfo[i][2])-y0)**2)#此处有问题，一直为0(已解决，用array列循环)
			dist = ((np.array(pICinfo[:,1]) - x0)**2 + (np.array(pICinfo[:,2])-y0)**2)**0.5
			#print(dist)
			self.totaCost.append((weighCol*dist).sum())
			for j in range(100):
				weighCol_n = weighCol/dist
				weighColX_n = ((np.array(pICinfo[:,1])*weighCol)/dist).sum()
				weighColY_n = ((np.array(pICinfo[:,2])*weighCol)/dist).sum()
				x,y = weighColX_n/weighCol_n.sum(),weighColY_n/weighCol_n.sum()
				dist = np.sqrt((np.array(pICinfo[:,1])-x)**2+(np.array(pICinfo[:,2])-y)**2)
				self.totaCost[i] = (weighCol*dist).sum()
				self.solu[i][0:2] = x,y
	def outputFunc(self):
		print("求解完毕")
		print("{}个待规划的物流中心的坐标如下：".format(self.k))
		print(self.solu)
		for i in range(self.k):
			print("第{}个物流中心的服务对象有：".format(i+1),end='')
			print(self.pIC[i][:,0])
			print("该物流中心的配送总成本为:{}".format(self.totaCost[i]))
if __name__ == '__main__':
	main = mulCOG()
	main.inputFunc()
	main.justKMeans()
	main.optSolu()
	main.outputFunc()
	main.showCluster()



