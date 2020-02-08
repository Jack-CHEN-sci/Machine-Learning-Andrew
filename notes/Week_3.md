### < Machine Learning > by Andrew Ng @ Stanford University

# Note of The 3rd Week's Lecture

## 逻辑回归（Logistic Regression）

### 分类（Classification）
+ 例子：
	+ 邮件：垃圾邮件/正常邮件
	+ 线上交易：诈骗/正常
	+ 肿瘤：恶性/良性
+ 分类问题就像回归问题一样，只是我们现在要预测的值只包含少量的离散值。
+ 典型的**二分类**（binary classification problem）任务：y∈{0，1}
	+ y=0时一般表示某种性质的缺失（Negative class）
	+ y=1时一般表示某种性质的存在（Positive class）
+ 分类任务不能用回归方法解决
	+ 回归方法：计算出线性回归函数后，映射所有大于0.5的预测作为1，所有小于0.5的预测作为0
	+ 回归方法并不适用，因为分类问题实际上不存在一个能够拟合的线性函数
+ 注意：逻辑回归实际上并非用于回归任务，而是用于分类任务

### 假设函数的表达式（Hypothesis Representation）
+ 需要改变假设函数h(x)的形式以将其输出范围限制在(0, 1)区间内
+ 做法如下：将theta^T*X放入一个特殊的函数中进行映射
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/sigmoid_function.png" width = 15% /></div>
+ 该函数称为**S型函数(Sigmoid Function)** 或**逻辑函数(Logistic Function)**，图像如下：
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/sigmoid_function_image.png" width = 70% /></div>
#### 假设函数表达的意义
+ h(x)将给出输出值为“1”的**概率**
	+ 例如：若h(x)=0.7，则输出值为“1”（具有某种性质）的概率为70% ，而输出值为“0”（不具有某种性质）的概率为30%
+ 数学表达即为：</br>
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/classification_hx_meaning.png" width = 40% /></br>
	其中，P(y=1|x;θ) 表示在给定假设函数的输入值x及其相关参数值θ后，输出值为1的概率</div>

### 决策边界（Decision Boundary）
+ 为了得到离散的0或1分类，我们可以将假设函数（Sigmoid）的输出转换如下：
	+ h(x)≥0.5 -> y=1 即 x≥0 -> y=1
	+ h(x)<0.5 -> y=0 即 x<0 -> y=0
+ 从图像上来看，假设函数会形成一条决策边界，边界一侧的数据点预测为“1”，另一侧预测为“0”
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/decision_boundary_image.png" width = 70% /></div>

### 代价函数（Cost Function）
+ 线性回归的代价函数在分类问题中并不适用
	+ 因为 1）分类问题中假设函数h(x)的形式很复杂； 2）我们对每一次预测产生的代价cost(h(x),y)的定义为预测与真实标签值之差的平方，平方之后无疑使得整个代价函数更为复杂
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/cost_redefine.png" width = 40% /></div>
	+ 事实上，若沿用线性回归的“均方误差”代价函数，则会导致代价函数J(θ)“非凸”（non-convex）
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/convex_vs_nonconvex.png" width = 70% /></div>
+ 定义适用于逻辑回归的预测代价
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/cost_of_classification.png" width = 60% /></div>
	+ 通过观察图像理解函数意义
		+ 当实际标签值y=1时：
			+ 若预测值 h(x) -> 0，则代价 cost -> ∞.（预测失误）
			+ 若预测值 h(x) -> 1，则代价 cost -> 0.（预测正确）
		+ 当实际标签值y=0时：
			+ 若预测值 h(x) -> 0，则代价 cost -> 0.（预测正确）
			+ 若预测值 h(x) -> 1，则代价 cost -> ∞.（预测失误）
	+ ***可以理解为我们对算法的预测失误进行处罚，失误越大，处罚越严厉***
+ 将新定义的 预测代价（cost）代入 代价函数（cost function）中，就可以保证 J(θ) 是凸函数了

### 简化代价函数
+ 将代价cost(h(x), y)由两种情况的分段函数写为统一形式:
+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/calssification_cost.png" width = 60% /></div>
+ 将上式代入代价函数J(θ)中，得到完整的代价函数表达式：
+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/calssification_cost_function.png" width = 60% /></div>
+ 在实际应用中，进行向量化之后的表达式为：
+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/vectorized_cost_function.png" width = 50% /></div>

### 梯度下降
+ 梯度下降的通式为：
+ <img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/gradient_descent_general.png" width = 22% />
+ 计算末尾的微分项后得到：
+ <img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/classification_gradient_descent.png" width = 35% />
+ 向量化后得到：
+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/vectorized_classification_gradient_descent.png" width =35% /></div>

### 高级优化算法
+ 优化算法：
	+ 给定代价函数J(θ)，欲求使得J(θ)最小的参数θ
	+ 算法执行过程中，对于当前的θ，需要计算:
		1. J(θ)的函数值
		2. J(θ)对各个θ_j (j = 0,1,...,n)的导数值
	+ 优化算法举例：
		+ 梯度下降
		+ Conjugate Gradient
		+ BFGS
		+ L-BFGS
	+ 高级优化算法不需要手动调试学习率α，并且通常执行较快；但是算法过程很复杂

### 多分类任务：一对多算法（Multiclass Classification: One-vs-all）
+ 举例：
	+ 邮件标签：工作，朋友，家庭，爱好，...
	+ 医疗诊断：无病，感冒，流感，...
	+ 天气：阴，晴，雨，雪，...
	+ 需要将预测类别从 y∈{0, 1} 二类拓展到 y∈{0, 1, ... , n} 多类
+ 假设函数的意义：某数据点属于该类的概率，如果将多分类任务划分为n个二分类任务，那么得到的n个假设函数分别就代表某数据点属于某一类的概率，取这n个概率值中最大的一个为该数据点的预测结果
+ 一对多算法：
	+ 以三类为例：
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/one_vs_all.png" width = 90% /></div>
	+ 数学表达：
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/multiclass_classification.png" width = 30% /></div>

## 正则化（Regularization）

### 过拟合
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/overfit_vs_underfit.png" width = 80% /></div>

+ **欠拟合**：算法具有“**高偏差 (High bias)**” —— 假设函数h(x)没有较好地拟合数据，该函数的与数据的趋势很不匹配,它通常是由于函数太简单或使用的特征（变量）太少造成的。
+ **过拟合**：算法具有“ **高方差 (High variance)** ” —— 假设函数h(x)对数据过度拟合，该函数虽然适合现有数据，但不能很好地 **泛化** 以预测新数据，它通常是由于函数太复杂造成的，会产生许多与数据无关的不必要的曲线和角度。
+ 解决过拟合问题有两个主要选项：
	1. 减少特征（变量）的数量
		+ 手动选择保留哪些特征
		+ 使用某种模型选择算法
	2. 正则化
		+ 保留所有特征，但减少参数θ的大小
		+ 当我们有很多具有微小作用的特征时，正则化将表现非常好

### 代价函数
+ 如果假设函数h(x)过拟合，我们可以通过增加代价函数来减少函数中某些变量（特征）的权重
+ 举个例子：我们想使一个四次函数（如下）形状更趋近于二次函数
	+ <img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/quartic_function.png" width = 35% />
	+ 那么我们想尽可能地弱化 x^3 和 x^4 的影响，在不去除这些特征，也不改变假设函数形式的前提下，我们可以修改代价函数:
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/regularized_cost_function_example.png" width = 65% /></div>
	+ 通过在代价函数中加上 θ_3 和 θ_4 相关且对代价函数值影响非常大的项，在降低代价的过程中，我们就必须缩小 θ_3 和 θ_4 ，即将 x^3 和 x^4 的权重降低
+ 通常情况下，我们不知道哪几个特征是需要被降权的，所以，我们可以在一个简单的求和中正则化所有的参数:
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/regularized_cost_function.png" width = 55% /></div>
	+ 其中，λ 是正则化参数，它决定了参数的代价扩大了多少
+ 利用上述带有额外一求和项的代价函数，我们可以使假设函数的形式更加平滑以减少过拟合
+ 注意：如果 λ 太大，它可能会使函数过于平滑并导致欠拟合。

### 正则化的线性回归
+ 代价函数
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/regularized_cost_function.png" width = 55% /></div>
+ 梯度下降
	+ <img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/regularized_linearReg_GD.png" width = 80% />
	+ 这里将 θ_0 与其他参数 θ_j (j = 1, 2, ... , n) 分开处理，我们将 θ_j 的式子变形如下：
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/regularized_linearReg_GD_transform.png" width = 55% /></div>
	+ 上式中等号右边第一项一般都是小于1的，这就起到了在每次迭代中缩小 θ_j 的作用。注意，此时等号右边第二项与正则化之前一样
+ 标准方程
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/regularized_linearReg_NE.png" width = 35% /></div>
	+ 方程的大致形式与原来的方程是一样的，只不过在括号里增加了另一项
	+ 注意：当 m<n 时，(X^T)X 是不可逆的；但是当我们加入 λL 后，(X^T)X+λL 就是可逆的了

### 正则化的逻辑回归
+ 代价函数
	+ <div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/regularized_logisticReg_costFunction.png" width = 85% /></div>
+ 梯度下降
	+ 梯度下降过程与线性回归是一致的，只是式中假设函数 h(x) 不同
	+ <img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/regularized_linearReg_GD.png" width = 80% />
