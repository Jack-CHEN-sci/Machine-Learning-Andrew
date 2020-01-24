### Machine Learning by Andrew Ng @ Stanford University

# Note of Lecture 1

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
	+ 例子：下棋
		+ T = “下棋”这项任务
		+ P = 程序赢得下一局的概率
		+ E = 下很多很多局棋后积累的经验


