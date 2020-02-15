### < Machine Learning > by Andrew Ng @ Stanford University

# Note of The 5th Week's Lecture

## 神经网络（Neural Network）

### 代价函数（cost function）
+ 规定以下记号：
	+ ***L***: 网络的总层数
	+ ***s_l***：第 *l* 层中的节点数量（除去偏差单元）
	+ ***K***：输出层的节点数（即类的数量）
	+ 因为神经网络可能有很多输出单元，我们记***h(x)_k***为假设函数结果的第k个值

+ 公式：
	+ 基于逻辑回归的代价函数
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/regularized_logisticReg_costFunction.png" width=80%></div>
	+ 改造为神经网络的代价函数
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/nn_cost_function.png" width=85%></div>
+ 对比：
	1. 参数量的增加，由一个假设函数的参数向量***θ*** ，到L-1层神经网络的参数矩阵***Θ*** 
	2. 公式的前半部分由一重累加和变为二重累加和，因为要将输出层中的K个单元的计算结果相加
	3. 公式的后半部分由一重累加和变为三重累加和，因为要将L-1层神经网络每层的参数矩阵的所有元素平方相加
	4. 注意：类似于逻辑回归，对常数项θ_0，我们想象为θ_0 * x_0，并且不对其进行正则化；在神经网络中对于每层参数矩阵的第0行第0列不进行正则化

### 反向传播算法（Backpropagation）
+ 注意：反向传播算法并非一个不同于梯度下降的算法，而是梯度下降算法运行在神经网络中时比较有效的计算方法
+ 反向传播算法（Backpropagation）包括前向传播（Forward pass）和后向传播（Backward pass）两个过程，分别承担不同的计算任务
+ 链式法则（chain rule）—— 反向传播算法的数学基础
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/chain_rule.png" width=45%></div>
+ 例子：（理解反向传播算法，李宏毅）
	+ 假设有如下神经网络架构：
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_example_nn.png" width=50%></div>
	+ 对其输入层与第二层第一单元之间的关系进行分析：
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_example_compute.png" width=50%></div>
	计算 **代价函数** ***C*** 对 **参数** ***w*** 的导数，根据链式法则，可拆解为计算 **函数** ***z(w)*** 对 **参数** ***w*** 的导数与 **代价函数** ***C*** 对 **变量** ***z*** 的导数之积。其中，第一部分可以在 **前向传播** 的过程中直接得到：函数 ***z(w)*** 对 参数 ***w*** 的导数即为前一层网络中与参数w相关的变量 ***x***。但是第二部分则需要在 **后向传播** 中计算
	+ 后向传播计算过程推导</br>
	考察下一层神经网络（假设所有神经元的激活函数均为sigmoid函数），如图：
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_example_compute1.png" width=70%></div>
	由链式法则，代价函数 ***C*** 对 变量 ***z*** 的导数可拆解为激活函数 ***a(z)*** 对 变量 ***z*** 的导数与代价函数 ***C*** 对 变量 ***a*** 的导数。其中，第一部分由激活函数的形式而确定；而第二部分，鉴于此神经元的计算结果（受参数w影响）参与到了下一层两个神经元的计算中，由链式法则，可以得出如图中最下方计算公式，简单整理后，在最右侧公式中，便可以发现递归计算的身影</br>
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_example_compute2.png" width=40%></div>
	通过以上推导过程，我们不难看出：类似于递归算法，若想计算代价函数对某一个参数（第1层中某个单元）的导数，必须先计算代价函数对第2层某个单元的导数，进而必须先计算代价函数对第3层某些(≥2)单元的导数，……，如此层层深入，先得出后面结果，否则无从计算前面。这就是 **后向传播** 的计算过程</br>
	+ 后向传播在网络中的“计算流”
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_example_nn1.png" width=50%></div>
	+ 小结：反向传播的两大过程 —— “前向传播”与“后向传播”
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_example_summary.png" width=50%></div>
+ 算法过程
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_algo.png" width=70%></div>
	+ 导入训练数据集，将神经网络每一层的偏差矩阵 ***∆^(l)*** 初始化为零矩阵
	+ 对于m个训练数据，依次进行：
		+ 将特征向量输入第一层网络（输入层）
		+ 执行前向传播，计算并保存沿途各层的输出结果 ***a^(l), l = 2, 3, ... , L***
		+ 计算网络输出与真实值的偏差向量 ***δ^(L)***
		+ 执行后向传播，计算并保存沿途各层的偏差向量 ***δ^(l), l = L-1, L-2, ... , 2***，向量化后得到如下式子：
			<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_delta_partial_compute.png" width=30%></div>
			<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_delta_vectorized.png" width=40%></div>
		+ 更新各层偏差矩阵 ***∆^(l)***，向量化公式如下：
			<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/backpropagation_deltaCap_compute.png" width=30%></div>
	+ 计算代价函数对参数矩阵的偏导数
		+ 矩阵 ***D^(l)*** 用作“累加器”，在计算的过程中把各个分部结果累加起来，最终计算出我们所求的偏导数

### 反向传播算法实践

#### 实现细节：参数展开（Unrolling Parameters）
+ 不同于线性回归和逻辑回归，当我们使用神经网络时，参数的数学组织形式不再是向量，而是矩阵
	+ *** Θ^(1), Θ^(2), Θ^(3), ... ***
	+ *** D^(1), D^(2), D^(3), ... ***
+ 为了能够使用 ***fminunc()*** 等高级优化算法，我们需要将以上矩阵“展开”成向量。例如，在Octave/MATLAB中使用以下命令：
	```
	thetaVector = [ Theta1(:); Theta2(:); Theta3(:); ]
	deltaVector = [ D1(:); D2(:); D3(:) ]
	```
+ 从“展开”后的向量变回矩阵（假设原矩阵Theta1的大小为10x11, Theta2的大小为10x11，Theta3的大小为1x11）：
	```
	Theta1 = reshape(thetaVector(1:110),10,11)
	Theta2 = reshape(thetaVector(111:220),10,11)
	Theta3 = reshape(thetaVector(221:231),1,11)
	```
+ 小节：
	<div align=center><img src="img/unroll_parameters.png" width=60%></div>

#### 实现细节：梯度检查（gradient checking）
+ 作用：检查神经网络反向传播算法的正确性 —— 是否正确计算梯度
+ 由梯度的几何意义（切线），代价函数J(Θ)对特定参数Θ的偏导数可以计算如下：
	+ 当只有一个参数矩阵时：
	<div align=center><img src="img/derivative_approximation.png" width=35%></div>
	+ 当有多个参数矩阵时，对Θ_j的偏导数：
	<div align=center><img src="img/derivative_approximation1.png" width=65%></div>
	+ 其中，***𝜖(epsilon)*** 为任意极小值，一般可以取10^(-4)，若太小会出现数值计算问题
+ Octave/MATLAB实现：
	```
	epsilon = 1e-4;
	for i = 1:n,
  		thetaPlus = theta;
  		thetaPlus(i) += epsilon;
  		thetaMinus = theta;
  		thetaMinus(i) -= epsilon;
  		gradApprox(i) = (J(thetaPlus) - J(thetaMinus))/(2*epsilon)
	end;
	```
+ 一旦我们计算出gradApprox，与之前计算得到的deltaVector进行比较，若 gradApprox ≈ deltaVector 则说明反向传播算法实现正确
+ 一旦验证了反向传播算法是正确的，就不需要再次计算gradApprox，因为gradApprox的计算代码运行很慢

#### 实现细节：随机初始化（random initialization）
+ 不同于线性回归和逻辑回归中将参数向量初始化为零向量，在神经网络中，参数矩阵不能初始化为零矩阵，因为这会使得隐藏层的各个单元输入均为零
+ 而对参数矩阵的其他形式的初始化也不一定行得通，例如下图中的初始化方式 —— 来自同一神经元的联结参数相等（在图中使用同种颜色表示），会导致后面所有神经元在反向传播时的计算结果相同，即所有神经元计算同一函数，捕捉同一特征，使得神经网络“空有其表”，形成巨大冗余
<div align=center><img src="img/nn_wrong_init.png" width=50%></div>
+ 因此，需要对神经网络的参数矩阵进行 **随机初始化**：将其赋值为区间[-𝜖,𝜖]中的随机数（注：此处的epsilon与梯度检查中的epsilon没有任何关系）
+ Octave/MATLAB实现：
	```
	% If the dimensions of Theta1 is 10x11, Theta2 is 10x11 and Theta3 is 1x11.
	% rand(x,y) is just a function in octave that will initialize 
	% 	a matrix of random real numbers between 0 and 1.
	
	Theta1 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
	Theta2 = rand(10,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
	Theta3 = rand(1,11) * (2 * INIT_EPSILON) - INIT_EPSILON;
	```

#### 实现小节
+ 首先，选择一个神经网络架构：共有多少层、每层有多少个神经元 ……
	1. 输入层单元数 = 特征向量的维度
	2. 输出层单元数 = 类别数量
	3. 隐藏层单元数： 通常情况下越多越好，但是需要权衡随着单元数量增加导致的计算成本增加
	4. 如果隐藏层的数量大于1，那么最好每层的神经元数量是相等的
+ 训练神经网络
	1. 随机初始化参数（权重）
	2. 执行前向传播，对每个特征向量计算其预测值
	3. 计算代价函数
	4. 执行后向传播，计算梯度/偏导数
	5. 使用梯度检查，确保反向传播算法执行正确，然后关闭检查
	6. 使用梯度下降算法或其他高级优化算法最小化代价函数得到参数值
