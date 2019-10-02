# Lecture 9: CNN Architectures

----------
## 목차
1. Case Studies
    - AlexNet
    - VGG
    - GoogleNet
    - ResNet
2. Other architectures to know…
    - NiN
    - Wide ResNet
    - ResNeXT
    - Stochastic Depth
    - DenseNet
    - FractalNet
    - SqueezeNet
    
----------
## Case Studies

2012년 CNN이 DL에 사용되면서 여러 CNN Architecture 들이 발표되었는데 이번 챕터에서는 각 architecture 들의 특징과 장단점에 대해서 살펴본다.


----------

**AlexNet**


![AlexNet](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1569997933121_+2019-10-02++3.32.11.png)


2012년 ImageNet 에서 비약적인 성능향상을 가져오면서 발표되었던 AlexNet 이다. CNN 아키텍쳐의 유행을 이끈 논문이며 가장 초창기의 모델이라고 할 수 있다.
특징을 보면 여러 conv layer 와 pool layer 를 이용하여 모델을 만들어내는 것을 볼 수 있다.

**Details / Retrospectives:**

- first use of ReLU
- used Norm layers (not common anymore)
- heavy data augmentation
- dropout 0.5
- batch size 128
- SGD Momentum 0.9
- Learning rate 1e-2, redueced by 10 manually when val accuracy plateaus\
- L2 weight decay 5e-4
- 7 CNN ensemble: 18.2% → 15.4%
----------

**VGG**


![VGG](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1569998176803_+2019-10-02++3.36.14.png)


기존 AlexNet 의 layer 갯수 (8)를 2배 이상 늘려서 만든 네트워크이다.
AlexNet 에서는 큰 conv layer를 사용하였었는데 VGG 에서는 작은 conv layer 를 사용하여 depth를 높혔다. 또한 parameter의 크기도 적어진다.

> 3x3 conv layer 를 3개 합치면 7x7 conv layer 와 같은 필드를 가지게 된다.
> Parameters : 3*(3^2 * C^2) vs 7^2 * C^2

**Details** :

- Similar training procedure as AlexNet
- No Local Response Normalisation (LRN)
- Use ensembles for best results
- FC7 features generalize well to other tasks


----------

**GoogleNet**


![GoogleNet](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1569998600637_+2019-10-02++3.43.18.png)


Network 는 더욱 Deep 해졌고 computational efficiency 는 높아졌다.
GoogleNet 은 network 안에 작은 network 를 만드는 것으로 구성되어진다. 이를 Inception Module이라고 칭하는거 같다.
Inception Module 은 Input 으로 들어온 데이터에 대하여 3가지의 conv layer 와 pool layer 를 통과한 결과를 concat 하여 다음 Inception Module 로 넘겨준다. (이게 왜 효과가 있는지는 사실 잘 모르겠다.)
이런 과정에서 굉장히 많은 computational cost 가 발생하는데 이를 줄이기 위해 bottleneck layer 를 추가하여 cost 를 줄인다.

![](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1569998879698_+2019-10-02++3.47.58.png)


이 과정에서 정보의 손실이 일어날 수 있으나 거시적인 관점에서는 문제가 안된다.(고한다)
여러 Inception Module 을 통과하면서 1개의 최종 classifier 와 2개의 auxiliary classifier 가 있는데 상호 소통하면서 loss 를 줄이는 방향으로 간다고 한다. 자세한건 설명하지 않음..

**Detail**:

- 22 layers
- Efficient “Inception” module
- No FC layers
- 12x less params than AlexNet


----------

**ResNet**


![ResNet](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1569999067205_+2019-10-02++3.51.04.png)


이전에는 8 개의 layer, 22 개의 layer 를 두면서 조금씩 변화했는데 이녀석은 152 개의 layer 로 갑자기 뻥뻥뻥튀기 하면서 나온 네트워크이다.
ResNet 의 Res 는 Residual 에서 나온 단어로 뜻은 나머지? 전차 이런식으로 해석된다

> output - input = 전차 란다… 

그전에도 단순히 layer 의 갯수를 늘리면 어떻게 되는지 실험은 되었다. 단순히 layer 를 늘리면 test error 가 layer 가 적당할때보다 높은 수치를 나타내었는데 이는 overfitting 을 의심할 수 있다. 하지만 test error 뿐만 아니라 training error 까지 높아져서 overfitting 의 문제가 아니라는 것을 알 수 있었다.

~~ResNet 을 만든 사람들은 optimization 의 문제라고 하는데 이건 하나의 가설~~

이 문제를 해결하기 위해서는 학습된 layer 을 shallow model 에서 가져와 identity mapping 으로 layer 를 추가해야 한다고 한다. ~~(먼말이냐)~~


![Residual](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1569999354420_+2019-10-02++3.55.49.png)


왼쪽의 그림은 일반적인 layer 이고 오른쪽은 Residual block 을 나타낸다.
Residual block 의 경우는 들어온 x 가 identity mapping 이 되어 conv layers 를 통과하고 더해주는 모습을 볼 수 있다. ( F(x) + x )

이러한 방식으로 넘겨줄 때 conv layers(F(x)) 들은 output으로 나오는 x (H(x)) 를 학습하는 것이 아니라 그저 x의 변화율을 학습( H(x) - x) 한다고 한다.
생각보다 간단한 테크닉이면서 심오한 내용이 담겨있다. 수학적으로 어떻게 되는지 궁금하다.

또한 50 개 이후의 layer 에서는 bottleneck layer 를 이용하여 efficiency 를 높힌다.


![](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1569999659879_+2019-10-02++4.00.57.png)


**Full ResNet architecture**:

- Stack residual blocks
- Every residual block has two 3x3 conv layers
- Periodically, double # of filters and downsample spatially using stride 2
- Additional conv layer at the beginning
- No FC layers at the end (only FC 1000 to output classes)

**Training ResNet in practice**:

- Batch Normalization after every CONV layer
- Xavier/2 initialization from He et al.
- SGD + Momentum (0.9)
- Learning rate: 0.1, divided by 10 when validation error plateaus
- Mini-batch size 256
- Weight decay of 1e-5
- No dropout used


----------



## Other architectures to know…



----------

**NiN**


![NiN](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1570000038262_+2019-10-02++4.07.16.png)


bottleneck layer 에 영감을 준 네트워크


----------

**Wide ResNet**


![Wide ResNet](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1570000263545_+2019-10-02++4.11.01.png)


ResNet 의 깊이는 중요하지 않다! 고 주장하면서 나온 개선 네트워크
Residual block 의 크기를 키워서 병렬화 연산을 진행하고 depth 를 줄였다 (computaional cost down)


----------

**ResNeXT**


![ResNeXT](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1570000384978_+2019-10-02++4.13.02.png)


GoogleNet 의 Inception module 을 ResNet 에 적용한 모습.
Residual block 의 width 를 여러 layer 로 넓혀서 parallel 동작을 하게 만듬.


----------

**Stochastic Depth**

![Stochastic Depth](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1570000472395_+2019-10-02++4.14.30.png)


Layer 의 깊이가 깊어질수록 초기 layer 의 gradient 가 vanishing 되는 것은 지난 슬라이드를 통해 알 수 있다. (chain rule 의 한계)
이 방법은 random 하게 layer 를 drop 시켜서 identity connection 을 만든다.
기존의 dropout 과 비슷함.


----------

**FractalNet**


![FractalNet](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1570001617718_+2019-10-02+16.33.21.png)


residual 은 필요 없다! 하면서 나온 네트워크
training 할 때 랜덤하게 path 를 정해서 간다고 하는데.. 잘모르겠닭


----------

**DenseNet**

![DenseNet](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1570001760553_+2019-10-02++4.35.59.png)


Input 을 block 안에서 여러번 concat 하는 방법.
이러한 방법을 쓰면 vanishing gradient 를 할 수 있다고 한다… 도대체 왜??? ㅜㅜ


----------

**SqueezeNet**


![SqueezeNet](https://paper-attachments.dropbox.com/s_8A515A44A0F9975E6C2EE56E2CA5D4C5718B40A00E3199DFE61E1102D2B223E7_1570002081974_+2019-10-02++4.41.19.png)


1x1, 3x3 conv 로 레이어를 squeeze 시킴.
그냥 AlexNet 과 비슷한 성능을 내지만 메모리를 510배정도 줄인 네트워크.
쓸모가 있는 네트워크인지는 모르겠지만 임베딩하거나 다른 네트워크를 만들때 아이디어를 가져올 수는 있겠다고 생각된다.


----------

Jun’s Note

1. 무엇을 배웠는지
    > 여러 convolutional architectures 들을 배우고 특징을 암
2. 어디에 쓰이는지
    > 이제 model 를 사용하면서 어떠한 모델을 사용할지 choose 하는 과정에서 아키텍쳐의 특징을 고려하여 선택가능
3. 수식의 목적을 잘 파악했는지?
    > 수식없다.

