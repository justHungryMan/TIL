# Lecture 8: Deep Learning Software

----------
## 목차
1. CPU vs GPU
    - Multiclass SVM loss
2. Deep Learning Frameworks
    - Caffe / Caffe2
    - Theano / TensorFlow
    - Torch / PyTorch
    
----------
## CPU vs GPU


----------

CPU : Central Processing Units
GPU : Graphical Processing Units

CPU 는 여러 연산을 지원하지만 GPU 는 간단한 수학 연산만 지원한다. 
CPU 는 2019년 기준 6코어 ~ 8코어 정도가 기본이 되지만 GPU 는 코어의 갯수가 기본 천개를 넘어간다.
GPU 자체에 메모리가 달려있어서 GPU Memory 에 데이터를 올리고 연산하는 방법이 최적화하는 방법중 하나가 될 수 있다.
Training 을 진행함에 있어서 data 를 SSD / HDD 에서 reading 하는 작업이 bottleneck 이 될 수 있다. 따라서 미리 메모리에 데이터를 올리는 등 코드를 짤때 최적화에 신경을 써야한다.


----------
## Deep Learning Frameworks

The point of deep learning frameworks

    - Easily build big computational graphs
    - Easily compute gradients in computational graphs
    - Run it all efficiently on GPU
----------

프레임워크에 관한 설명이므로 내용은 주요 내용은 적지 않겠다.

**Static vs Dynamic Graphs**

![](https://paper-attachments.dropbox.com/s_7F64C0F4809D19D3312EB735EE3158A910A9DC8DB866A004F4428E4F09E74EBA_1569987956659_+2019-10-02++12.45.54.png)


tensorflow 의 경우 computational graph 가 Static 하게 정의되고 수행된다. 
pytorch 의 경우 computational graph 가 Dynamic 하게 정의되고 수행된다. 마치 파이썬 스럽게…


----------

Jun’s Note

1. 무엇을 배웠는지
    > CPU와 GPU의 차이점, 딥러닝 프레임워크
2. 어디에 쓰이는지
    > 딥러닝 프레임워크를 사용할 때 프레임워크들이 어떠한특징을 가지는지 알 수 있음
3. 수식의 목적을 잘 파악했는지?
    > 수식없어.. ㅎ

