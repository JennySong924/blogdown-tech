---
title: PLS 偏最小二乘回归
author: ' J Song'
date: '2024-08-27'
slug: pls
---
> 引自生瓜蛋子[博客](https://blog.csdn.net/qq_51320133/article/details/137503042)

## 引言与背景
偏最小二乘回归（Partial Least Squares Regression，PLSR）是一种统计学和机器学习中的多元数据分析方法，特别适用于处理因变量和自变量之间存在多重共线性问题的情况。该方法最早由瑞典化学家 Herman Wold 于上世纪 60 年代提出，作为一种多变量线性回归分析技术，广泛应用于化学、环境科学、生物医学、金融等领域，尤其在高维数据和小样本问题中表现出色。

## PLS 定理与原理
偏最小二乘回归核心思想是通过寻找新的正交投影方向（主成分），使得投影后的自变量和因变量之间有最大的协方差，进而建立预测模型。不同于主成分回归 PCR 单纯地对自变量进行降维，PLSR 在降维过程中同时考虑了因变量和自变量的相关性，以期在降低维度的同时最大化预测性能。

## 算法原理
PLS 算法分为以下步骤：
- 提取主成分：首先计算自变量和因变量的协方差矩阵，通过迭代算法（如 NIPALS 算法）提取出第一组主成分，这组主成分既能反映自变量的变化趋势，又能反映因变量的变化趋势。
- 回归建模：将提取出的主成分作为新的自变量，对因变量进行线性回归建模
- 重复迭代：对剩余自变量残差继续提取新的主成分，并进行回归，直到满足预定的停止准则（如累计解释变异率达到设定阈值，或提取的主成分数目达到预设值）。

## 算法实现
- python
```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

pls = PLSRegression(n_components=2)
pls.fit(X_train,y_train)

y_pred = pls.predict(X_test)

r2 = r2_score(y_test,y_pred)
```

- R
```r
library(pls)

pls.model <- plsr(y~x,data,validation='CV')
# Find the number of dimensions with lowest cross validation error
cv <- RMSEP(pls.model)
best.dims <- which.min(cv$val[estimate = "adjCV", , ]) - 1
 
# Rerun the model
pls.model <- plsr(y~x, data, ncomp = best.dims)
```

## 优缺点
- 优点：
  - 处理多重共线性：自变量之间高度相关时，也可以通过提取主成分进行有效回归分析
  - 高维数据处理：PLS 能够降维提炼信息并构建预测模型
  - 小样本下表现优良：在样本数量较少的情况下仍然能够获得较为理想的预测效果，因为它强调的是变量之间的关系而非样本量
  
- 缺点：
  - 过拟合风险：在主成分数量选择不当（如过多）时，可能会导致过拟合
  - 非线性关系处理能力有限
  - 参数敏感性：PLS 中参数设置（如主成分的数量）对模型解释性、预测性能、防止过拟合有很大影响，需要根据实际问题和数据特点进行反复试验和调整
  
  
## 与其他算法对比
- OLS：OLS 在自变量之间存在多重共线性时，参数估计会出现问题，例如标准误差增大、参数估计失当等。PLS 既能解决多重共线性问题，又保留了预测性能，在处理这类数据时优于 OLS。
- PCR：主成分回归仅关注自变量的降维，PLS 不仅考虑了自变量之间的关系，同时也考虑了因变量与自变量之间的关联，使得提取的主成分能同时优化因变量预测性能，所以在预测时 PLS 一般能取得更好的效果
- SVM和决策树等非线性模型：擅长处理非线性关系，PLS 则更擅长处理线性关系

