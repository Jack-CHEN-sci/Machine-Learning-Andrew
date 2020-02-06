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
