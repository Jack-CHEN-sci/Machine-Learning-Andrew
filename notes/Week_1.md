### < Machine Learning > by Andrew Ng @ Stanford University

# Note of The 1st Week's Lecture

## 欢迎
### 机器学习
+ 来自于人工智能的研究中
+ 使得计算机具备了新的能力

### 应用例子
+ 数据挖掘
	+ 从自动化系统或网络中采集到的大型数据集
	+ 如：网页中的点击信息，药物记录，生物信息，工程信息等
+ 无法“手动”编程的应用
	+ 如：自动驾驶直升机，手写文字识别，大部分的自然语言处理（NLP），计算机视觉
+ “定制化”编程
	+ 如：Amazon, Netflix产品推荐
+ 理解人类的学习过程

## 概述
### 定义
+ Arthur Samuel (1959)：一个研究如何使计算机在没有被明确编程的情况下具备学习能力的研究领域

>A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P, if its performance at tasks in T, as measured by P, improves with experience E. （Tom Mitchell）

+ Tom Mitchell (1998)： 当计算机程序在执行 ***任务T*** 时的 **性能** （由 ***度量P*** 来描述）随着 ***经验E*** 的积累而逐渐增长，那么就可以说该程序从 ***经验E*** 中 **学习** 到了某些类型的 ***任务T*** 以及 ***性能度量P*** 。
	+ 例子：垃圾邮件过滤
		+ T = 对邮件进行分类：spam or not spam
		+ P = 正确分类的邮件数量
+ 机器学习算法：
	+ 监督学习
	+ 无监督学习
	+ 其他：强化学习、推荐系统……
### 监督学习
+ 特点：“正确答案（输出）”已被给出
+ 回归任务：预测连续变量
+ 分类任务：预测离散变量
### 无监督学习
+ 特点：“正确答案（输出）”是未知的，需要根据数据集中变量之间的关系寻找分布规律
+ 聚类任务：在大量数据中寻找能将其划分为多个不同组别的特征
+ 非聚类任务：“鸡尾酒会问题”——从嘈杂的环境中寻找结构

## 一元线性回归

### 一些记号
+ m：训练样本规模
+ x：输入变量/特征
+ y：输出变量/目标变量
+ (x,y)：一个训练样本
+ h：hypothesis（假设）—— 算法学习到的函数

### 代价函数
+ 用代价函数来衡量假设函数h的准确性：它取假设函数h对输入x的所有结果h(x)与和实际输出y的平均值之差(实际上是平均值的更fancy的版本)。
+ **平方误差函数**（squared error function）或**均方误差**（mean squared error）。

<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/mean_squared_error.png" width = 50% /></div>

+ 式中取均值后再乘以1/2，是因为它更够使得梯度下降的计算更加方便，因为平方式求导后的系数会将其抵消。

### 梯度下降
+ 过程类似：下山（二维情况，如下图）。在当前位置环视四周，寻找当前最快的下山路径，不断重复。最终，山的最低点就是代价函数的最小值

<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/gradient_descent.png" width = 50% /></div>

+ 做法：对代价函数求导，该点的导数就是山的坡度，也就是移动的方向。代价函数沿着当前最陡的下降方向移动，移动的步长通过参数\alpha —— **学习率**（learning rate）—— 决定。
+ 学习率决定下降的步长，代价函数的导数决定下降的方向
+ 不同起始点出发后最后到达的最低点可能不同
+ 算法：

<p style="text-indent:20em"> repeat until convergence:</p>
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/gradient_descent_algo.png" width = 30% /></div>
<p style="text-indent:20em"> where j=0,1 represents the feature index number.</p>

+ 注意：对于每一轮迭代，参数的更新应该是同步的：在计算完毕后再统一更新；在有其他参数未完成计算之前更新某一已完成计算的参数会造成错误

<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/gradient_descent_implement.png" width = 50% /></div>
