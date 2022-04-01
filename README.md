# naic_2021_CSI_Feedback
2021年全国人工智能大赛 “AI+无线通信” 赛道  
初赛 Score：72.5  Rank 25   复赛Score：0.9970  Rank15  

信道数据来源于多个复杂场景下采样得到的真实无线信道信息，数据集包含 10,000 个信道数据样本，覆盖多个复杂场景，每个场景包含若干样本。每个样本是一个 126*128 的二维 CSI 矩阵（可以把单个样本视为一张图片），其中 126 代表时延抽头数目，128 代表天线数目（32 发 4 收）。每个场景内的 CSI 样本具有一定的特征相关度。  

网络结构如下：

![Aaron Swartz](network.png)
