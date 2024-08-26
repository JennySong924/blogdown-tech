---
title: Lasso
author: J Song
date: '2024-08-26'
slug: lasso
---
1. R code
```{r}
library(glmnet)

cv_model <- cv.glmnet(x, y, alpha = 1)
best_lambda <- cv_model$lambda.min
best_model <- glmnet(x, y, alpha = 1, lambda = best_lambda)
coef(best_model)
```

2. 应用性质
- 变量个数可以超过样本量
- 在 R 中跑 Lasso 时，函数不会给出 R2，Lasso 的目标也不是最大化 R2，如果需要计算 R2 得单独写两行代码
