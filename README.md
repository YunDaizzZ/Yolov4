# Yolov4
这是一个yolov4_pytorch代码 


没有实现所有trick，做了以下实现：

backbone：DarkNet53 => CSPDarkNet53

特征金字塔：SPP、PAN

训练：Mosaic数据增强、标签平滑、学习率余弦退火衰减、CIOU

激活函数：Mish
