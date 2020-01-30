### < Machine Learning > by Andrew Ng @ Stanford University

# Note of The 2nd Week's Lecture

## 多元线性回归

### 一些记号
+ m：训练样本规模
+ n：变量/特征/维度数
+ x^(i)：第i个训练样本的特征向量
+ x^(i)_j：第i个训练样本的特征向量的第j个特征
+ h：hypothesis（假设）—— 算法学习到的函数

### 回归函数
+ 多元线性回归拟合（假设）函数形式：
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/multiple_features_function.png" width = 50% /></div>

	+ x_1, x_2, ... , x_n 表示n个变量（特征）
	+ theta_0, theta_1, ... , theta_n 为需要计算的参数</br></br>

+ 对任意 i ∈ 1,…,m，x^(i)_0 = 1. 那么将上式向量化后写为：
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/multiple_features_function_vector.png" width = 50% /></div>

###  梯度下降算法
+ 算法实质与一元线性回归一致，表达形式近乎一致：</br>
	<img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/multiple_features_GDAlgo.png" width = 60% />
	+ 累加的是m个训练样本（特征向量）在当前预测函数h(x)下的预测值与实际值之差的平方对第j个变量（特征）的导数
	+ 全部n+1个参数theta计算完成后更新
+ 将其各项逐个写出即为：</br>
	<img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/multiple_features_GDAlgo(1).png" width = 45% />
