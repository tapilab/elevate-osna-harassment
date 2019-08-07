---
layout: slide
title: "LogisticRegression"
---

Parameters changed: penalty, C

'penalty': Used to specify the norm used in the penalization.

'C':Inverse of regularization strength; must be a positive float. Like in support vector machines, smaller values specify stronger regularization.


|  penalty | Accuracy |
|----------|----------|
|    l1    |   0.72   |
|    l2    |   0.72   |


|     C    | Accuracy |
|----------|----------|
|    .1    |   0.70   |
|    1.0   |   0.72   |
|    5.0   |   0.70   |
|   10.0   |   0.70   |


| F1-score |Precision |  Recall  |
|----------|----------|----------|
|   0.72   |   0.72   |   0.72   |
