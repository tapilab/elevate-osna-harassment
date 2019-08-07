---
layout: slide
title: "MLPClassifier"
---

Parameters changed: hidden_layer_sizes, alpha

|hidden_layer_sizes| Accuracy |
|------------------|----------|
|      (10, )      |   0.69   |
|      (50, )      |   0.69   |
|      (100, )     |   0.71   |
|      (200, )     |   0.69   |


|   alpha  | Accuracy |
|----------|----------|
|   .001   |   0.69   |
|  .0001   |   0.71   |
|  .00001  |   0.69   |


| F1-score |Precision |  Recall  |
|----------|----------|----------|
|   0.66   |   0.66   |   0.65   |
