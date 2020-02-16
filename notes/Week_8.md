### < Machine Learning > by Andrew Ng @ Stanford University

# Note of The 8th Week's Lecture

## 无监督学习（Unsupervised Learning）
+ 无监督学习输入的数据集是没有标签的（因缺乏足够的先验知识，而难以人工标注类别或进行人工类别标注的成本太高），我们希望算法能够代替人类找出数据中的内在结构，例如：聚类
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/supervised_VS_unsupervised.png" width=80%></div>
+ 无监督学习的应用举例：
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/apps_of_unsupervised_learning.png" width=70%></div>

## K均值算法(K-means Algorithm)
+ 在聚类问题中，我们有未加标签的数据，我们希望有一个算法能够自动的把这些数据分成有紧密关系的子集或簇
+ K均值算法是现在最为广泛使用的聚类方法
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/Kmeans_example.png" width=100%></div>
+ 算法概况：
	+ 输入: 
		+ ***K***（计划从数据中聚类出的簇的个数）
		+ 无标签训练集 ***{x^(1), x^(2), ... , x^(m)}***
	+ **Initialization** : 随机初始化K个点，这K个点叫做 **聚类中心** (cluster centroids)
	+ 迭代以下步骤：
		+ **Cluster Assignment (簇分配)** : 遍历所有的样本，依据每一个点是更接近哪个聚类中心来将每个数据点分配到不同的簇中
		+ **Move Centroids (移动中心)** : 分别计算每个簇的中心（均值），将聚类中心移动至各个簇的中心位置
+ 算法伪代码：
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/Kmeans_algo.png" width=70%></div>

### 优化目标函数（Optimization Objective）
+ 失真代价函数（distortion cost function）
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/Kmeans_optimize_objective.png" width=50%></div>
+ 从算法过程上来看：
	+ 簇分配环节：固定 ***μ_1, ... ,μ_K*** 不变，调整 ***c^(1), …,c^(m)***，使得代价函数最小
	+ 移动中心环节：固定 ***c^(1), …,c^(m)*** 不变，调整 ***μ_1, ... ,μ_K***，使得代价函数最小

### 随机初始化（Random Initialization）
+ 推荐的随机初始化方法：随机从训练集中选取K个数据点作为聚类中心
+ 问题：受制于不同的初始化聚类中心，最终的聚类结果可能不同，最差的情况是算法会陷入局部最优解，如图:
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/Kmeans_init_optima.png" width=80%></div>
+ 为防止聚类算法陷入局部最优解，可以采用多次尝试取最优的策略，即采用多个（50 ~ 100）不同的初始化聚类中心分别得到最优解，比较取最小失真代价的聚类结果
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/Kmeans_multi-init.png" width=65%></div>

## 降维（Dimensionality Reduction）

### 简介
+ 以低维情况为例：
	+ 在二维平面中，若两个变量 ***x_1, x_2*** 的线性相关性很强，那么我们可以假设一个新的变量 ***z_1*** ，将二维数据映射为一维数据
	+ 在三维空间中，若数据点从总体上看大致分布在某一个平面上下，那么我们可以假设两个新的变量 ***z_1, z_2*** 构成二维平面，将三维数据映射到该二维平面上
	+ 注：通常，在使用机器学习算法的问题情境中，数据的维度可达到1000+甚至更多，降维的幅度更大
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/DR_example.png" width=80%></div>
+ 降维的好处：
	1. **数据压缩**</br>
	 使用降维技术，我们不仅保留了数据本来的分布特点，并且减少了数据的维度，降低了存储空间。将降维运用到机器学习算法的数据中，可以很大程度上提升算法运行效率
	2. **可视化**</br>
	 使用降维技术，我们可以将无法在低维空间中直接表达的高维数据展示在屏幕上，直观地感受数据分布特点

### 主成分分析（Principal Component Analysis）
