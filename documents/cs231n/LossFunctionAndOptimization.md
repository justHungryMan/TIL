# Lecture 3: Loss Functions and Optimization

----------
## 목차
1. Loss Function
    - Multiclass SVM loss
    - Cross-entropy loss (Softmax)
2. Optimization
    - Gradient Descent
    - Stochastic Gradient Descent
    

Loss Function 이란 training data에 대하여 score가 얼마나 불만족스러운지를 판단하는 기준이다. 이러한 Loss 값을 낮추기 위하여 parameter를 조절하는데 이를 optimization 이라고 한다.

Dataset 의 $$x_i$$ 를 image라 하고 $$y_i$$ 를 label이라 하자.
$$\{(x_i, y_i)\}_{i=1}^N$$

이때 Loss Function을 나타내면 다음과 같다.

이는 적당한 함수를 이용하여(f) 각 이미지의 Loss 값을 구하고 이를 평균낸다.

**Multiclass SVM loss**
SVM을 나타내면 다음과 같다.

**Cross-entropy loss (Softmax)**

