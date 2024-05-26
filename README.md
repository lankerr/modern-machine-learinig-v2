# 本大作业主要是使用MNIST来实现生成式的四大模型:
# 1. GAN
# 2. VAE
# 3. DE
# 4. DIFFUSION

主要的内容包括在project里面了

# 一个奇怪的bug
在写文件的时候，我必须同时有/home/huiln/project和/project/两个文件夹都需要放文件和模型
# 修复
/home/huilin是绝对路径，home/huilin是相对路径，所以在写文件的时候，我全部修改为绝对或者相对就可以了

我个人认为是os和torch之类的库用的链接一个是绝对的，一个是相对的，所以会出现这个问题

# 失败
将MNIST迁移到CAFRI10数据集上之后，是失败的，失败原因我暂时不清楚，我暂时认为是模型过于简单，encoder和decoder过于复杂，然后训练的时候没有收敛

# 修复

我注意到使用MNISTFashion数据集的时候，我使用的可以训练，我突发奇想降低latent_dim试了一下，这可以，应该是mnist数据集的问题，我尝试将latent_dim降低到4，8，然后训练，这次可以，所以我认为是latent_dim的问题，并且在latent=16的时候突然不能使用，但是在latent=4，8，32的时候可以使用，所以我认为是一些随机收敛的问题，可以进行进一步研究

# 一定的比对
这里因为模型deep-energy有一个打分的系统，采用deep-energy为这些模型打一个分数，然后比对这些模型的效果

最后结果:diffusion>VAE>GAN>DE，这里是基于deep-energy-based    score函数的打分，得到的结果
主要我觉得还是GAN的模型定义过于粗糙，而diffusion基于郎之万的模型，VAE基于变分推断，DE基于能量函数，所以在一定数学假设的前提下模型的效果会非常的显著

