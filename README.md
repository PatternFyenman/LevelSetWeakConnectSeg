Taichi Hackathon

团队名：羊羊队
项目名：基于水平集方法的弱连接像素区域分割
选题方向：图像处理

# 项目介绍
---
## 创作背景

在图像分割任务中，相连区域的目标分割是一个重要的研究内容。区域之间可能存在多种相连方式，比如两个目标部分区域重叠或互相贴合、两个目标之间存在像素强度较弱的连接。本项目针对后者（弱连接像素区域），基于taichi语言，使用水平集方法分割弱连接相连目标。下图展示一种弱连接目标，图片来源HPAIC。

![[00c9a1c9-2f06-476f-8b0d-6d01032874a2_yellow(crop).png]]


水平集方法是通过隐式地表达高维曲面，用基于曲率的速度函数F驱动水平集函数φ在低维空间上实现形状的合并和分离。在xyz坐标系中，将三维空间中的水平集函数在二维平面z=0上的剖面轮廓，称为零水平集。水平集函数被初始化后，可以根据特定的演化方程逐渐移动，零水平集随之移动到目标边界，从而将目标分割出来。下方视频展示了基于水平集方法的图像分割过程。在演化结束时，视频左侧上方为三维空间中的水平集函数，左侧下方为移动到目标边界的零水平集。水平集方法在许多人工合成和自然图像上取得了令人满意的目标分割结果，但是，当前的水平集方法还未能实现弱连接目标的细胞图像分割。
![[水平集方法分割图像示例.mp4]]

## 实现方法

本项目在三维空间实现二维图像中相连目标的分割问题。水平集函数主要是在x-y二维平面方向上移动，但少有人考虑让水平集在z方向上移动。当水平集函数在抵达弱连接的两个目标区域边界后，使水平集函数在z方向移动，到两个弱连接目标的连接部分。然后基于物理切割模型，让水平集在z方向上切割弱连接部分，从而实现弱连接区域的分割。在三维空间中，将二维图像的像素强度看成是山的高度，两个弱连接的区域就像是两座大山，两座山之间有低矮的山谷将两座山相连，水平集函数如同一把开山斧劈向山谷，将两座山分离开。

见下图，考虑x-y二维图像的某一剖面图，中间蓝色为x方向或y方向像素强度分布，绿色由浅到深表示随迭代次数增加水平集函数的位置。虚线表示水平集函数抵达深绿色的目标轮廓之后，断开两个弱相连区域。

![[Pasted image 20221123093049.png]]
