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
	+ 对于参数theta_j的计算：累加的是m个训练样本（特征向量）在当前预测函数h(x)下的预测值与实际值之差的平方和对第j个变量（特征）的导数
	+ 全部n+1个参数 theta_0, theta_1, ... , theta_n 计算完成后再更新
+ 将其各项逐个写出即为：</br>
	<img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/multiple_features_GDAlgo(1).png" width = 45% />

### 梯度下降的加速方法（tricks）
#### 特征缩放
+ 将输入的数值规范为同样大小的规模，避免梯度下降过程中的反复震荡</br>
+ 将输入值除以输入变量的范围(即最大值减去最小值)，得到的新范围在-1到1之间。
<img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/feature_scaling.png" width = 90% />
+ 作用：通过减少梯度下降的迭代次数从而起到加速作用
	+ 注：不能解决梯度下降陷入局部最优解的问题
#### 均值归一化
+ 从每一输入变量的值中减去全体输入变量的平均值，从而得到一个平均值为0的新输入变量。
+ 遵循如下公式：</br>
	<img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/mean_normalization.png" width = 20% /></br>
	式中ui为所有变量i的值的平均值，si为变量i的值的范围（max-min）或标准差（standard deviation）

### 怎样选择合适的“学习率α”？
+ **调试梯度下降**：以迭代次数为x轴，代价函数J(theta)的值为y轴画出变化图。如果J(theta)一直在增长，那么应当将学习率减小
+ **自动收敛判断**：若代价函数J(theta)在一次迭代中减少的值小于E（E是一个很小很小的值，例如0.001），那么可以判定J(theta)收敛了。但是在实际应用中选择合适的阈值E是非常困难的
+ 已经证明：如果学习率α足够小,那么J(θ)将在每次迭代中都减少

## 自定义特征与多项式回归
+ 我们可以将多个特征组合为一个，例如将特征x1, x2通过乘法运算x1*x2组合为新特征x3
### 多项式回归
+ 若线性回归无法较好的拟合数据的分布，那么假设函数h(x)就需要是非线性的
+ 可以通过将假设函数h(x)变成二次函数、三次函数或平方根函数(或任何其他形式)来改变曲线形状
+ 注意：特征缩放！
	+ 平方和开方运算会较大地改变特征（变量）的取值范围
	+ 例如，x1∈(1,1000)，那么平方后x1^2∈(1,1000000)，立方后x1^3∈(1,1000000000)，开方后√x1∈(1,32)

## 标准方程
+ 与梯度下降的作用一样，用于寻找使得代价函数J(theta)最小的theta值
+ **标准方程**（normal equation）提供了一种高效的方法，可以直接解出theta
+ 与梯度下降不同，标准方程不是一个迭代算法，而是通过分别对theta_j（每一个j）求导数并令其等于0求解得到。方程如下：</br>
	<img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/normal_equation.png" width = 20% /></br>
	其中，X为m×(n+1)的矩阵，m行：每行为某个样本的特征向量，n+1列：n个特征 + 1个常数项因子（“1”）
+ 使用标准方程的话无需对特征进行缩放

### 梯度下降VS标准方程
|梯度下降|标准方程|
|:-:|:-:|
|需要选择学习率α|无需对学习率α进行选择|
|多次迭代|没有迭代过程|
|复杂度为O(k*n^2)|复杂度为O(n^3)，需要计算矩阵(X^T)*X的逆|
|算法在n非常大时表现稳定|如果n非常大，那么算法执行时间会非常长|
+ 注：在实践中，当特征数n超过10,000时，可以考虑从标准方程法转为使用梯度下降。

### 不可逆性
+ 线性代数中，并非所有矩阵都是可逆的，我们将可逆矩阵称为奇异（singular）矩阵或退化（degenerate）矩阵
+ Octave中
	+ 函数pinv()：pseudo-inverse 伪逆（一般使用此函数可以得到想求的theta）
	+ 函数inv()：inverse 逆
+ 什么情况下X^T*X不可逆？
	1. 存在冗余特征（线性相关的）
		+ 例如：在预测房价时，特征x_1为房子的面积，单位为feet^2，特征x_2为房子的面积，单位为m^2
		+ 因为1m = 3.28feet，故x_1 将恒等于 (3.28)^2 * x_2
	2. 过量特征（e.g. 样本数量 ≤ 特征数量）
		+ 解决方案：删除一部分特征 或 使用正则化（regularization）的方法
