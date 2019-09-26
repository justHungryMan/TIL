# Lecture 7: Training Neural Network 2

----------
## 목차
1. Fancier Optimization
    - Optimization : Problem with SGD
    - SGD + Momentum
    - Adam
2. Regularization
    - Model Ensembles
    - Dropout
    - Data Augmentation
3. Transfer Learning


    
----------
## Fancier optimization


----------

**Optimization : Problem with SGD**

SGD 를 이용하여 최적의 loss 값을 찾는 과정을 optimization 하는 과정을 살펴보자.

![local minima (상) saddle point (하)](https://paper-attachments.dropbox.com/s_E05F066AA161AD2EECD3880B22282E18DE7861D15212BD4442D3489440663E70_1569464521834_+2019-09-26++11.21.59.png)


SGD 를 이용한다면 local minima 에 빠지거나 saddle point 에 빠지는 경우가 생각보다 빈번하다. (특히 강의에서는 saddle point 에 정말 많이 빠진다고 한다.) 
SGD 를 이용할때 minibatches 를 이용하기 때문에 노이즈도 많이 발생할 수 밖에 없다.
이를 해결하기위해 현실에서 호리병 모양의 통에 공을 굴리면 가장 낮은 지점에 가듯이 gradient descent 과정에서 속도의 값을 반영하여 최적의 loss 값을 찾는 방법이 있다.


----------

**SGD + Momentum**
****
기존의 SGD 에서는 w 의 값을 learning weight * gradient 를 뺀 값으로 update 하였는데 이는 최적의 loss 로 향하는 방향으로 업데이트 하는 의미로만 볼 수 있다. 여기서 기존의 최적의 loss 로 향하는 속도를 추가해주면 우리가 원하는 방향으로 갈 수 있다.

![SGD vs SGD + Momentum](https://paper-attachments.dropbox.com/s_E05F066AA161AD2EECD3880B22282E18DE7861D15212BD4442D3489440663E70_1569464908108_+2019-09-26++11.28.25.png)


이러한 방법으로 weight 를 update 해주면 local minima 와 saddle point 에서도 멈추지 않고 계속 진행되는 것을 알 수 있다. 


----------

**Adam**

가장 많이 쓰이면서 좋은 momentum 방법이다.

    fisrt_moment = 0
    second_moment = 0
    for t in range(num_iterations):
      dx = compute_gradient(x)
      first_moment = beta1 * first_moment + (1 - beta1) * dx
      second_moment = beta2 * second_moment + (1 - beta20 * dx * dx
      first_unbias = first_moment / (1 - beta1 ** t) # Bias correction
      second_unbias = second_moment / (1 - beta2 ** t)
      x -= learning_rate * first_unbias / (np.sqrt(second_unbias) + 1e - 7))


----------
## Regularization


----------

**Model Ensembles**
****
여러개의 독립적인 모델을 만들어내고 이들의 결과를 테스트 할 때 평균내어 사용하는 방법이다.
이 방법을 사용하면 2%정도의 성능 향상을 기대할 수 있다.

모델 앙상블을 하는 과정에서 여러개의 독립적인 모델을 만드는 방법 외에도 하나의 모델을 training 도중 여러개의 snapshot 을 만들어 활용하는 방법도 있다.

![Model Ensembles](https://paper-attachments.dropbox.com/s_E05F066AA161AD2EECD3880B22282E18DE7861D15212BD4442D3489440663E70_1569468776128_+2019-09-26++12.32.53.png)

----------

**Regularization**

Regularization 기법은 train 과 validation 을 할 때 사용되는 데이터의 accuracy 와 test 하는 데이터의 accuracy 간의 gap 을 좁히기위해 사용된다.

![](https://paper-attachments.dropbox.com/s_E05F066AA161AD2EECD3880B22282E18DE7861D15212BD4442D3489440663E70_1569468988001_+2019-09-26++12.36.25.png)



----------

**Dropout**

Dropout 은 forward pass 를 진행할 때 random 하게 뉴런들을 없애는 작업을 의미한다.


![](https://paper-attachments.dropbox.com/s_E05F066AA161AD2EECD3880B22282E18DE7861D15212BD4442D3489440663E70_1569469325967_+2019-09-26++12.42.03.png)


이러한 방법이 의미가 있다고 판단되는 이유는 학습을 진행할때 판단되는 무수히 많은 조건들 (예를 들어 고양이는 갈색이다, 발톱이 있다, 귀가 머리위에 있다 등)을 어느정도 줄여줘서 그렇지 않을까 생각이 든다. (이에 대한 다양한 해석이 있다.)

Dropout 을 한 네트워크를 test 하는 과정을 살펴보자.
학습이 진행되는 동안 뉴런들이 없어질 가능성을 test 타임 때 생각해야 한다.
Test time 때는 모든 뉴런들이 항상 활성화 상태이다. 따라서 모든 뉴런들을 forward pass를 진행할때 고려하였던 확률을 가중치로 하여 뉴런에 곱연산을 해준다.

![](https://paper-attachments.dropbox.com/s_E05F066AA161AD2EECD3880B22282E18DE7861D15212BD4442D3489440663E70_1569470116519_+2019-09-26++12.55.14.png)



- 그런데 보통 mini-batch 를 이용하면 dropout 에서 생길 문제점이 해결된다.


----------

**Data Augmentation**

같은 사진을 이용하여 여러 사진을 만든다면 regularization 에도 도움을 줄 것이다.
예를들어 고양이의 사진이 있다면 그 사진을 좌우 반전시켜도 고양이는 고양이여야 한다.


![Color Jitter](https://paper-attachments.dropbox.com/s_E05F066AA161AD2EECD3880B22282E18DE7861D15212BD4442D3489440663E70_1569470586586_+2019-09-26++1.03.04.png)



----------
## Transfer Learning

Transfer Learning 은 기존의 CNN 을 학습하기 위해선 엄청난 양의 데이터와 시간이 필요한데 이를 해결하기 위해 고안된 방법이다.

![](https://paper-attachments.dropbox.com/s_E05F066AA161AD2EECD3880B22282E18DE7861D15212BD4442D3489440663E70_1569470730849_+2019-09-26++1.05.28.png)


비슷한 데이터에 대해서 모델을 다시 학습시킬 필요는 없으니 마지막에 classification 해주는 FC layer 만 수정하는 방식이다.
이러한 방법은 생각보다 대중적이여서 최근에 나오는 모델들은 대부분 transfer learning 기법을 이용한다.


----------

Jun’s Note

1. 무엇을 배웠는지
    > Neural Network 를 만들때 부가적으로 학습률을 높힐 수 있는 방법들을 배웠다.
2. 어디에 쓰이는지
    > Neural Network 를 구성할 때
3. 수식의 목적을 잘 파악했는지?
    > 사실 Momentum 과 Dropout 에 나오는 수식을 제대로 이해하지 못했다. Adam 방법에서는 왜 기존의 값을 저런 식으로 update 하는지 의문이다.

