活动名：Taichi Hackathon

团队名：羊羊队

项目名：基于水平集方法的弱连接像素区域分割

选题方向：图像处理

# 项目介绍

## 创作背景

在图像分割任务中，相连区域的目标分割是一个重要的研究内容。区域之间可能存在多种相连方式，比如两个目标部分区域重叠或互相贴合、两个目标之间存在像素强度较弱的连接。本项目针对后者（弱连接像素区域），基于taichi语言，使用水平集方法分割弱连接相连目标。下图展示一种弱连接目标，图片来源[HPAIC](https://www.proteinatlas.org/humanproteome/subcellular)。

![image](https://github.com/PatternFyenman/LevelSetWeakConnectSeg/blob/main/README_media/example1.png)


水平集方法是通过隐式地表达高维曲面，用基于曲率的速度函数F驱动水平集函数φ在低维空间上实现形状的合并和分离。在xyz坐标系中，将三维空间中的水平集函数在二维平面z=0上的剖面轮廓，称为零水平集。水平集函数被初始化后，可以根据特定的演化方程逐渐移动，零水平集随之移动到目标边界，从而将目标分割出来。下方视频展示了基于水平集方法的图像分割过程。在演化结束时，视频左侧上方为三维空间中的水平集函数，左侧下方为移动到目标边界的零水平集。水平集方法在许多人工合成和自然图像上取得了令人满意的目标分割结果，但是，当前的水平集方法还未能实现弱连接目标的细胞图像分割。


https://user-images.githubusercontent.com/82517756/203458117-a1a02881-9698-4154-91ed-66ba9b9fdebd.mp4



## 实现方法

本项目在三维空间实现二维图像中相连目标的分割问题。技术路线见下图。

![image](https://github.com/PatternFyenman/LevelSetWeakConnectSeg/blob/main/README_media/flow%20chart.png)

水平集函数主要是在x-y二维平面方向上移动，但少有人考虑让水平集在z方向上移动。当水平集函数在抵达弱连接的两个目标区域边界后，使水平集函数在z方向移动到两个弱连接目标的连接区域。然后基于物理切割模型，让水平集在z方向上切割弱连接部分，从而实现弱连接区域的分割。在三维空间中，将二维图像的像素强度看成是山的高度，两个弱连接的区域就像是两座大山，两座山之间有低矮的山谷将两座山相连，水平集函数如同一把开山斧劈向山谷，将两座山分离开。

见下图，考虑x-y二维图像的某一剖面图，中间蓝色为x方向或y方向像素强度分布，绿色由浅到深表示随迭代次数增加水平集函数的位置。虚线表示水平集函数抵达深绿色的目标轮廓之后，断开两个弱相连区域。

![image](https://github.com/PatternFyenman/LevelSetWeakConnectSeg/blob/main/README_media/example3.png)

# 项目实现
目前代码实现了当确定模糊连接区域中某个像素点时，可以对该区域进行分割的功能，并且基于taichi框架，将原DRLSE水平集分割模型的运行时间减少了近70%。

## 分割的基本原理
在分割区域时，水平集函数值小于0的取值区域为目标区域，等于0的区域是目标轮廓，大于0的区域是目标外部的区域。

当水平集模型迭代收敛到两个模糊连接的目标轮廓附近时，可以在模糊连接区域中的某个位置（这个位置称为“切割点”），使水平集函数值小于0，然后继续运行水平集模型的迭代收敛程序。模糊区域中小于0的水平集函数会向外扩散，最终与两个模糊相连区域的其余轮廓合并，由此实现模糊区域分割的效果。

## 如何确定切割点？
第一种是人为确定切割点，专业生物医学研究人员根据研究内容确定哪个模糊连接区域需要分割，这在实际应用中十分常见。

第二种是根据水平集函数三维曲面法向量的散度决定。因为水平集函数在迭代收敛时，会逐渐包裹住两个模糊相连的区域，形成类似相连山峰的形状。在模糊连接部分，水平集函数曲面的曲率、法向量的方向和大小会出现较大的变化。计算水平集函数三维曲面法向量的散度，可以定位包裹模糊连接区域的零水平集的凹点位置，进而可以推算模糊连接区域中心的切割点位置。见下图，红圈部分亮度较高的点是散度值较高的点，根据这两个点可以推算中央区域的点。

![Divergence](https://user-images.githubusercontent.com/82517756/205471032-3ee7f30e-5a16-4754-8f3c-ca36a6faa1ca.png)


## 如何切割？
这里展现了水平集方法的一个优势。如果确定了在何处切割，只需要把该处的水平集函数减小到小于0即可，然后让水平集依照之前的演化方程继续收敛，使零水平集合并。

在代码中，具体的做法是以切割点为中心，设定一个sigma为2的二维高斯函数，将切割点附近的区域减去4倍的二维高斯函数值（倍数4是自定义的）。

## 如何加速水平集模型的演化
使用taichi中的空间稀疏数据结构。

在每次迭代时，只激活并计算整幅图像中水平集函数梯度大于1的区域，并在最外层循环时把算法置于taichi scope中对计算区域的每个点进行并行计算。

最终，使用taichi升级的DRLSE水平集模型，运算时间从原本的40s降低到13s左右，消耗的计算时间降低了70%！

感谢Yuanming Hu团队的辛勤付出，感谢taichi！

# 运行程序指南
我的运行环境：
![running_environment](https://user-images.githubusercontent.com/82517756/205471867-52697bd5-15c3-4f9b-9186-e29fcde09799.png)


我的python版本是3.10.6，确保您的计算机已pip install以下第三方库。
- matplotlib==3.6.2
- numpy==1.23.5
- opencv_python==4.6.0.66
- scikit_image==0.19.3
- scipy==1.9.3
- skimage==0.0
- taichi==1.2.2
- tqdm==4.64.1

在命令行运行如下指令，看看未使用taichi时对于1024x1024大小的图像，DRLSE模型迭代400次所需要的时间。
```shell
$ python3 DRLSE.py
```
在命令行运行如下指令，看看使用taichi后对同样大小的图像，在相同迭代次数时DRLSE模型的运行时间。
```shell
$ python3 taichi-DRLSE.py
```
在命令行运行如下指令，看看在确定切割点的情况下，水平集函数分割模糊连接点的过程。
```shell
$ python3 ManualCut.py
```
在命令行运行如下指令，看看使用水平集函数的曲面法向量的散度确定模糊连接区域的效果。（时间仓促，基于散度确定切割点的算法还不完善，后续会继续改进）
```shell
$ python3 DivergenceCut.py
```

