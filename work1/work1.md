# GPU架构与编程 课程大作业（一）

> 课程：中国科学院大学计算机学院专业选修课 GPU架构与编程
> 授课：赵地（中科院计算所）
> 学期：2026年秋季学期
> 截止日期：10月28日，截止日期之前可多次提交
> 提交要求：每位同学单独提交、单独评分

## 总体要求
基于CUDA语言，实现CNN推理，并在Fashion‑MNIST数据集集合上测试，**不能调用英伟达公司library**；
完成调优工作，对比英伟达官方库，评估自己实现版本的性能水平。

- 数据集网址：https://github.com/zalandoresearch/fashion-mnist
- 参考资料：[Accelerating GPU Applications with NVIDIA Math Libraries](https://developer.nvidia.com/blog/accelerating-gpu-applications-with-nvidia-math-libraries/)

## 问题一(基本题)：推理
### 任务
基于CUDA语言，实现CNN推理，不限制具体实现的网络，**推荐LeNet**。

需要实现模块：
- 卷积
- 池化
- 激活
- 与网络相关的其他模块

### 考核信息
- 考核指标：准确率，运行时间
- 考核平台：计算中心H20平台
- 考核人员：助教

### 评分标准

| 档次 | 分值 | 考核依据 |
|------|------|----------|
| 第一档 | 30（满分） | 准确率、运行时间 |
| 第二档 | 25 | 准确率、运行时间 |
| 第三档 | 20 | 准确率、运行时间 |


## 问题二(竞赛题)：在NutShellGPU上实现
### 任务
在NutShellGPU上实现CNN推理；
网络需要与问题一相同；鼓励运用学到的GPU知识，改进NutShellGPU，对推理代码进行极致优化。

需要实现模块：
- 梯度下降
- 与网络相关的其他模块

数据集：需要与问题一(基本题)相同
代码要求：需要提交代码进行考核

NutShellGPU仓库：shturl.cc/ilvZcWYRJXxo4njhDB9q4ovn7AG

### 考核信息
- 考核指标：准确率，运行时间
- 考核平台：计算中心 H20
- 考核人员：助教
- 评分：按排名计分
