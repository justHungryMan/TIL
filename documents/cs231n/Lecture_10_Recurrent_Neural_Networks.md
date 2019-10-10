# Lecture 10: Recurrent Neural Networks

----------
## 목차
1. RNN
    - RNN Architecture
    - Computational Graph
    - Backpropagation
    - Application
2. Advance
    - LSTM
    - GRU
    
----------
## RNN

RNN 은 글, 유전자, 손글씨, 음성 신호, 센서가 감지한 데이타, 주가 등 배열(sequence, 또는 시계열 데이터)의 형태를 갖는 데이터에서 패턴을 인식하는 인공 신경망 이다.

----------

**RNN Architecture**


RNN 은 기본적으로 다음 그림과 같이 hidden state 를 둬서 원하는 결과를 뽑아내는 구조이다.

![](https://paper-attachments.dropbox.com/s_B2B873FC9BA46CF380237409373F874275CD301FB93D9CE112C2A1A0D828A8C9_1570687310725_+2019-10-10++3.01.48.png)


각 hidden state 는 recurrence formula 형태를 띄고 있으며 state 는 계속 이전 state 에 영향을 받아 변화한다.

![Vanilla recurrence formula](https://paper-attachments.dropbox.com/s_B2B873FC9BA46CF380237409373F874275CD301FB93D9CE112C2A1A0D828A8C9_1570687492133_+2019-10-10++3.04.50.png)





----------

**RNN: Computational Graph**


![RNN : Computational Graph](https://paper-attachments.dropbox.com/s_B2B873FC9BA46CF380237409373F874275CD301FB93D9CE112C2A1A0D828A8C9_1570689112452_+2019-10-10++3.31.49.png)


Computational Graph 를 그리면 다음과 같다. 특이한 점은 각각의 hidden state 에서  결과값을 뽑아내어 Loss 를 구하고 이를 sum 해주는 것이다. 이렇게하면 backpropagation 을 할 때 add operator 로 인해 각각의 loss 에 대한 gradient 도 구하기 쉬울 것이다.
문제는 W이 모든 hidden state 에서 사용된다는 점….
이걸 해결하는 방법은 Advance Section 에서 다루겠다.



----------

**Backpropagation**

![](https://paper-attachments.dropbox.com/s_B2B873FC9BA46CF380237409373F874275CD301FB93D9CE112C2A1A0D828A8C9_1570689552919_+2019-10-10++3.39.10.png)


RNN 의 특징은 sequence 형태의 데이터를 이용하는 알고리즘이다보니 backpropagation 을 하려면 연산해온 모든 state 를 거쳐야한다. 이러한 방법을 사용하는 경우 시간이 너무 오래걸려 대안을 이용한다.



![Truncated Backpropagation](https://paper-attachments.dropbox.com/s_B2B873FC9BA46CF380237409373F874275CD301FB93D9CE112C2A1A0D828A8C9_1570690087858_+2019-10-10++3.48.05.png)


이 방법은 들어오는 sequence 의 길이가 길어도 일정단위로 자르고 gradient step 을 진행한다. 마치 SGD 와 비슷한 모양..

[쉬운 예제](https://gist.github.com/karpathy/d4dee566867f8291f086)

----------

**Application**


![Image Captioning](https://paper-attachments.dropbox.com/s_B2B873FC9BA46CF380237409373F874275CD301FB93D9CE112C2A1A0D828A8C9_1570692808998_+2019-10-10++4.33.26.png)



----------
## Advance


----------

**LSTM**

RNN 을 사용할 때 weight 가 vanishing 되거나 exploding 되는 현상을 막기 위해 만들어진 architecture

![LSTM](https://paper-attachments.dropbox.com/s_B2B873FC9BA46CF380237409373F874275CD301FB93D9CE112C2A1A0D828A8C9_1570693134219_+2019-10-10++4.38.52.png)


~~참 대단하다.. 딥러닝의 시대가 오기전에 나온 논문이라니..~~
위의 식을 보면 c_t 의 값을 backpropagation 하는 과정에서는 w 를 건들이지 않는다.

[좋은 블로그 참고](https://dgkim5360.tistory.com/entry/understanding-long-short-term-memory-lstm-kr)



----------

**GRU**

![](https://paper-attachments.dropbox.com/s_B2B873FC9BA46CF380237409373F874275CD301FB93D9CE112C2A1A0D828A8C9_1570694066377_+2019-10-10++4.54.24.png)


LSTM 을 변형시킨 것, 그러나 성능은 비슷하다고 한다.
GRU 는 input gate 와 output gate 를 update gate 로 합친 것과 같다고 하였다.


----------

Jun’s Note

1. 무엇을 배웠는지
    > RNN 과 변형 아키텍쳐
2. 어디에 쓰이는지
    > 연속성이 있는 데이터를 처리할때 사용하기 좋다.
3. 수식의 목적을 잘 파악했는지?
    > LSTM 의 수식이 무엇을 의미하는지 정확하기 파악하기 어렵다.


