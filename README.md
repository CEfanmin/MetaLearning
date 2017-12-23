# MetaLearning
对于ICML2017的paper《Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks》的算法理解与应用。

	核心算法如下图所示：
   ![multi](./algorithm.png "algorithm")

   内层和外层要同时优化的目标是权值参数Theta，本质上其实是学习一个很sensitive的权值（可以和迁移学习进行比较），使得能够**quickly (in a small number of gradient steps) and efficiently (using only a few examples)**。如下图所示：
   ![multi](./meta-ml.png "meta-ml")

