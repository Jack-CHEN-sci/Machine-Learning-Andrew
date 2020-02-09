### < Machine Learning > by Andrew Ng @ Stanford University

# Note of The 4th Week's Lecture

## 神经网络（Neural Network）
### 模型表达
+ 简单来讲，神经元（neurons）就是计算单元，它将输入模块（树突）获取电信号输入（称为“spikes”），并将其引导到输出模块（轴突）
+ 在我们的模型中：
	+ 轴突就是输入的特征 x_1, x_2, ... , x_n，树突输出的就是假设函数的计算结果
	+ 输入单元 x_0 被称作“**偏差单元**”（bias unit），其值总是“1”
	+ 仍使用在逻辑回归中使用过的 S型(sigmoid)函数 作为假设函数，在神经网络中，通常称为“S型**激活函数**”（sigmoid activation function）
	+ 参数θ通常也被称为“**权重**”
+ 节点与层
	+ 所有输入节点构成神经网络的第一层，通常称为“**输入层**”（Input layer）
	+ 最终输出假设函数结果的节点构成神经网络的最后一层，通常称为“**输出层**”（output layer）
	+ 输入层与输出层之间的层叫做“**隐藏层**”（hidden layers）
+ 举个例子：
	+ 下面这张图中，我们将隐藏层中的节点记作a，上标代表该节点处于第几层，下标代表该节点是该层的第几个，并将其称为“**激活单元**”（activation units）
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/one_hidden_layer_example.png" width = 25% /></div>
	+ 其中，各激活单元进行的是如下计算：（其中 ***Θ*** 表示相关的参数矩阵，上标表示该矩阵用于第几层的计算，下标表示矩阵的第几行第几列）
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/one_hidden_layer_computation.png" width = 55% /></div>
	+ 在本例中，我们用一个 3（该层中除偏差单元以外的激活单元数目）× 4（上一层中包含偏差单元的所有单元数目） 的参数矩阵来计算激活节点，将每一行的参数应用到输入中，以获得一个激活节点的值
	+ 假设函数的输出值取决于 1) 上一层激活节点值之和 与 2) 激活函数

### 向量化
+ 下面，我们将继续用上面所举的例子，对计算式进行向量化：
	+ 新定义向量 ***z*** ，上标代表节点所在层，下标表示节点在该层中的序号
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/one_hidden_layer_example_replace.png" width = 14% /></div>
	+ 也就是说，向量 ***z*** 意味着：
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/vector_z.png" width = 40% /></div>
	+ 当向量 ***x*** 是隐藏（激活）层 ***a*** 时，可将上式写为通式：
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/vector_z_rewrite.png" width = 20% /></div>
	+ 最后，输出层：
	<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/neural_network_hypothesis_vectorized.png" width = 27% /></div>

### 例子
#### 用简单神经网络模拟逻辑门
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/simple_nn_individual_example.png" width = 90% /></div>
#### 用多层神经网络模拟复杂逻辑进行分类
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/simple_nn_mixed_example.png" width = 90% /></div>

### 多分类
在多分类任务中，我们有多个输出单元，最终输出的向量长度等于类别数量，且向量中只有一个元素为“1”，代表预测的类别，其他元素为“0”
<div align=center><img src="https://raw.githubusercontent.com/Jack-CHEN-sci/Machine-Learning-Andrew/master/notes/img/nn_multiple_output_oneVSall.png" width = 90% /></div>
