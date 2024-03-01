---
layout: post
title:  "Unsupervised Scalable Representation Learning review"
date:   2024-03-1 15:13:23 +0900
categories:
  - Time Series
  - Paper Review
---
# "Unsupervised Scalable Representation Learning for Multivariate Time Series"

원문: <a href="https://proceedings.neurips.cc/paper/2019/hash/53c6de78244e9f528eb3e1cda69699bb-Abstract.html">Franceschi, Jean-Yves, Aymeric Dieuleveut, and Martin Jaggi. "Unsupervised scalable representation learning for multivariate time series." Advances in neural information processing systems 32 (2019).</a> 

{% include toc %}

간단하게 요약하자면 triplet loss를 쓰고 self supervised 하기 위한 positive negative sampling을 anchor를 기준으로 안은 positive 다른 time series에서 얻은건 negative로 정의한 논문으로입니다. 논문 리뷰를 시작하겠습니다.

# &lt; Introduction &gt;

이 연구에서는 시계열에 대한unsupervised general-purpose representation learning이라는 주제를 조사합니다. 자연어 처리(Young et al., 2018) 또는 동영상(Denton & Birodkar, 2017)과 같은 분야에서 representation learning에 대한 연구가 증가하고 있음에도 불구하고, 비시간적 데이터에 대한 구조적 가정 없이 시계열에 대한 범용 표현 학습을 명시적으로 다루는 논문은 거의 없습니다. 이 문제는 여러 가지 이유로 실제로 어려운 문제입니다.

1. 레이블이 거의 없다
    1. unsupervised representations learning이 강하게 선호된다
2. 다른 길이의 time series의 input도 허용하면서 compatible representation을 하는 방법론이여야한다. 
3. training과 inference 할때 scalability 와 efficiency가 주요하다. 
    1. 짧은 시간의 시계열 그리고 긴 시간의 시계열 둘다 작동을 잘 해야한다. 

따라서 본 논문에서는 연구된 시계열의 길이가 다양하고 잠재적으로 긴 문제에 부합하는 다변량 시계열에 대한 general-purpose representations을 학습하는 unsupervised 방법을 제안합니다. 이를 위해 우리는  scalable encoder를 훈련하는 **새로운 unsupervised loss를 소개**한다. 이 scalable encoder는 dilated convolutions이 포함된 deep convolutional nerual network의 모양을 가지고 fixed-length vector representations을 출력한다.  이 손실은 time-based negative sampling을 사용하는 **triplet loss로 구축**되어 길이가 같지 않은 시계열에 대한 인코더의 복원력을 활용합니다. 우리가 아는 한, 이것은 **시계열 관련 문헌에서 완전히 최초의 fully unsupervised triplet loss**입니다. 다양한 데이터 세트에서 학습된 표현의 품질을 평가하여 보편성을 보장합니다. 특히, UCR 저장소에 수집된 시계열 문헌의 표준 데이터 세트에 대한 분류 작업에 우리의 표현을 어떻게 사용할 수 있는지 테스트합니다(Dau et al., 2018). 우리는 우리의 표현이 일반적이고 전이 가능하며, 우리의 방법이 동시 비지도 방법보다 성능이 뛰어나고 심지어 non-ensemble supervised classification techniques의 최신 기술와도 일치한다는 것을 보여줍니다. 또한, UCR 시계열은 독점적으로 단변량이고 대부분 짧기 때문에, 우리는 최근 UEA 다변량 시계열 저장소(Bagnall et al., 2018)와 매우 긴 시계열을 포함한 실제 데이터 세트에 대한 표현을 평가하여 분류를 넘어 다양한 작업에 대한 확장성, 성능 및 일반화 능력을 입증합니다. 이 논문은 다음과 같이 구성되어 있습니다. 섹션 2에서는 시계열에 대한 비지도 표현 학습, 삼중 손실, 딥 아키텍처에 관한 선행 연구를 개괄적으로 설명합니다. 섹션 3에서는 인코더의 비지도 학습에 대해 설명하고, 섹션 4에서는 후자의 아키텍처에 대해 자세히 설명합니다. 마지막으로 섹션 5에서는 방법을 평가하기 위해 수행한 실험 결과를 제공합니다.

# &lt; Related Work &gt;

<div style="font-size: 20px; margin: 5px 0;"><strong>Unsupervised learning for time series</strong><br></div>
우리가 아는 한, 비디오나 고차원 데이터를 다루는 연구(Srivastava 외., 2015; Denton & Birodkar, 2017; Villegas 외., 2017; Oord 외., 2018)를 제외하고 시계열에 대한 unsupervised representation learning을 다룬 최근 연구는 거의 없습니다. Fortuin 등(2019)은 시계열의 변화를 잘 나타내는 시계열의 시간적 표현을 학습함으로써 이 작업과 유사하지만 다른 문제를 다루고 있습니다. Hyvarinen & Morioka(2016)는 시계열의 균등한 크기의 세분화에 대한 표현을 학습하여 이러한 표현에서 이러한 세분화를 구별하는 방법을 학습합니다. Lei 등(2017)은 학습된 표현 간의 거리가 시계열 간의 표준 거리(동적 시간 왜곡, DTW)를 모방하도록 설계된 비지도 방법을 노출합니다. (2017)은 인코더를 순환 신경망으로 설계하고, 디코더와 함께 학습된 표현으로부터 입력 시계열을 재구성하기 위해 시퀀스 간 모델로 공동 훈련합니다. 마지막으로, Wu 등(2018a)은 신중하게 설계되고 효율적인 커널의 근사치에서 생성된 특징 임베딩을 계산합니다. 그러나 이러한 방법은 either are not scalable nor suited to long time series(순환 네트워크의 순차적 특성 또는 입력 길이에 대한 이차적 복잡성을 가진 DTW 사용으로 인해), 표준 데이터 세트가 없거나 매우 적고 공개적으로 사용 가능한 코드가 없는 상태에서 테스트되거나 학습된 표현의 품질을 평가하기에 충분한 비교를 제공하지 못합니다. 확장 가능한 모델과 광범위한 분석은 이러한 방법을 능가할 뿐만 아니라 이러한 문제를 극복하는 것을 목표로 합니다.

<div style="font-size: 20px; margin: 5px 0;"><strong>Triplet losses</strong><br></div>
삼중 손실은 최근 다양한 영역에서 표현 학습을 위해 다양한 형태로 널리 사용되어 왔으며(Mikolov 외., 2013; Schroff 외., 2015; Wu 외., 2018b), 이론적으로도 연구되어 왔습니다(Arora et al, 2019), 오디오를 제외한 시계열에는 많이 사용되지 않았으며(Bredin, 2017; Lu et al., 2017; Jansen et al., 2018), 기존 연구에서는 학습 데이터에 클래스 레이블이나 주석이 있다고 가정하기 때문에 우리가 아는 한 fully unsupervised setting에서는 사용되지 않았습니다.보다 구체적인 다른 작업에 초점을 맞추고 있지만 우리의 작업과 유사한 Turpault 등(2019)은 반지도 환경에서 오디오 임베딩을 학습하면서 훈련 데이터의 특정 변환에 부분적으로 의존하여 삼중 손실에서 양성 샘플을 샘플링하고, Logeswaran & Lee(2018)는 무작위로 선택한 문장 중에서 다른 문장의 실제 문맥을 인식하도록 문장 인코더를 훈련하는데, 이는 시계열에 적응하기 어려운 방법입니다. 대신 저희는 **서브샘플링을 통해 유사성을 학습하여 보다 자연스러운 양성 샘플 선택에 의존**합니다.

<div style="font-size: 20px; margin: 5px 0;"><strong>Convolutional networks for time series</strong><br></div>
심층 컨볼루션 신경망은 최근 시계열 분류 작업에 성공적으로 적용되어 경쟁력 있는 성능을 보여주고 있습니다(Cui et al., 2016; Wang et al., 2017). 오디오 생성에 널리 사용되는 Dilated convolutions은 WaveNet(Oord et al., 2016)에서 성능을 개선하는 데 사용되었으며, 우리에게 영감을 준 아키텍처를 사용하여 시계열 예측을 위한 시퀀스 간 모델로서도 우수한 성능을 보였습니다(Bai et al., 2018). 이러한 연구는 특히 Dilated convolutions이 효율성과 예측 성능 측면에서 순환 신경망보다 우수한 순차적 작업용 네트워크를 구축하는 데 도움이 된다는 것을 보여줍니다.

# &lt; Unsupervised Training &gt;

Malhotra et al. (2017)가 수행한 decoder와 공동으로 학습할 필요성을 피하고 encoder-only arichitecture인 autoencoder-based standard representation learning methods을 학습하려고 하는데 이는 더 큰 계산 비용을 유발하기 떄문이다. 이를 위해, 저희는 성공적인 고전적인 단어 표현 학습 방법인 word2vec(Mikolov et al., 2013)에서 영감을 받아 시계열에 대한 새로운 triplet loss을 도입했습니다. 제안된 triplet loss은 original time-based sampling strategies을 사용하여 레이블이 지정되지 않은 데이터에 대한 학습의 어려움을 극복합니다. 우리가 아는 한, 이 연구는 시계열 문헌에서 완전한 비지도 환경에서 삼중 손실에 의존하는 최초의 연구입니다. 

목표는 유사한 시계열이 유사한 표현을 얻도록 하는 것이며, 이러한 유사성을 학습하기 위한 감독 없이도 가능합니다. 삼중 손실은 유사한 시계열이 유사한 표현을 얻도록 하는 것을 달성하는것에는 도움이 되지만(Schroff et al., 2015), 유사한 입력 쌍을 제공해야 하므로 감독없이 학습하기 것에 어려움이 있습니다.삼중 손실을 사용하는 시계열에 대한 이전의 지도 작업은 데이터가 주석이 달렸다고 가정하지만, 우리는 비지도 시간 기반 기준을 도입하여 유사한 시계열 쌍을 선택하고 다양한 길이의 시계열을 고려하여 word2vec의 직관에 따라 선택합니다. 

word2vec의 cbow모델에서 가정하는 것은 두가지 fold가 있다. The representation of the context of a word should probably be, on one hand, close to the one of this word (Goldberg & Levy, 2014), and, on the other hand, distant from the one of randomly chosen words, since they are probably unrelated to the original word’s context. The corresponding loss then pushes pairs of (context, word) and (context, random word) to be linearly separable. This is called negative sampling.

![image](https://github.com/passion3659/passion3659.github.io/assets/89252263/6f590192-0879-4012-b551-8c643afd725b)

이 원칙을 time series에 적용하기 위해서 우리는 random으로 subseries를 consiter한다. 그리고 x ref에서 뽑은 subseries는 x pos로 가까워야한다(a positive example). 그리고 다른 series에 있는 y를 neg로 뽑는다. 시간은 완전랜덤 상관없다(a negative sample). 그리고 이 negative sample은 x ref와 거리가 있어야한다. word2vec의 비유를 따라가자면, x pos는 word가 되고, x ref는 word가 속하는 context 그리고 x neg는 random 단어이다. 학습된 표현의 실험 결과뿐만 아니라 훈련 절차의 안정성과 수렴을 개선하기 위해 word2vec에서와 같이 여러개의 네거티브 샘플을 도입합니다.

일반적인 경우 알고리즘 1처럼 negative samples를 램덤으로 chose한다. 하지만 negative sample이 x pos랑 같은 size가 될수도 있다. 이럴때는 모든 dataset이 equal lenght를 가지고 있을때 적합할 것이다. 그리고 computation factorizations로 인해 훈련 절차의 속도가 빨라집니다. 알고리즘1은 dataset이 같은 길이를 가지고 있지 않을때 사용한다. 실험에서는 그냥 랜덤으로 했다. 

![image](https://github.com/passion3659/passion3659.github.io/assets/89252263/c54a8735-64ab-4151-8973-25475acfa66c)

이 시간 기반 삼중 손실은 선택한 인코더의 기능을 활용하여 다음과 같이 취할 수 있다는 점을 강조합니다. 입력 시계열을 취할 수 있다는 점을 강조합니다. 다양한 입력 길이 범위에서 인코더를 훈련함으로써 입력 길이의 범위에서 1부터 훈련 세트에서 가장 긴 시계열의 길이까지 다양한 입력 길이에 대해 인코더를 훈련시킴으로써 의미 있고 입력 길이에 관계없이 전달 가능한 표현을 출력할 수 있습니다(섹션 5 참조). 이 훈련 절차는 긴 시계열에 대해 실행할 수 있을 만큼 효율적이라는 점에서 흥미롭습니다. (섹션 5 참조) 확장 가능한 인코더(섹션 4 참조) 덕분에 디코더가 필요 없는 설계와 손실의 분리 가능성 덕분에 손실의 분리 가능성 덕분에 메모리를 절약하기 위해 용어당 역전파를 수행할 수 있습니다.

#  &lt; Encoder Architectur &gt;

이 섹션에서는 시계열에서 관련 정보를 추출해야 하고, 학습과 테스트 모두에서 시간 및 메모리 효율적이어야 하며, 가변 길이 입력을 허용해야 한다는 세 가지 요구 사항에 따라 선택한 인코더 아키텍처에 대해 설명하고 제시합니다.

우리는 시계열을 처리하기 위해 deep neural networks with exponentially dilated causal convolutions을 사용하기로 결정했습니다. 이 구조는 context of sequence generation에서 대중화되었지만(Oord et al., 2016), unsupervised time series representation learning에는 사용된 적이 없습니다. 이 방법은 몇 가지 장점이 있습니다. rnn계열과 다르게 GPU와 같은 최신 하드웨어에서 효율적으로 병렬화할 수 있어 확장성이 뛰어납니다. exponentially dilated convolutions은 receptive field를 exponential하게 증가시켜서 long-range dependencies를 더 잘 포착한다. 

컨볼루션 네트워크는 순차적 데이터에 대해서도 다양한 측면에서 성능이 우수하다는 것이 입증되었습니다. 예를 들어, 순환 네트워크는 반복적인 특성으로 인해 그라디언트가 폭발하고 소멸하는 문제가 있는 것으로 알려져 있습니다(Goodfellow et al., 2016, 10.9장). 이 문제를 해결하고 장기 종속성을 포착하는 능력을 향상시키기 위해 LSTM(Hochreiter & Schmidhuber, 1997)과 같은 많은 연구가 진행되었지만, 순환 네트워크는 여전히 이 측면에서 컨볼루션 네트워크보다 성능이 떨어집니다(Bai et al., 2018). 실험적 평가의 필수적인 부분인 시계열 분류와 예측의 특정 영역에서 심층 신경망은 최근 성공적으로 사용되었습니다(Bai et al., 2018; Ismail Fawaz et al., 2019).

(2018)에서 영감을 받아 인과적 컨볼루션, 가중치 정규화(Salimans & Kingma, 2016), 누수 ReLU 및 잔여 연결의 조합으로 네트워크의 각 계층을 구축합니다(그림 2b 참조). 이러한 각 레이어에는 기하급수적으로 증가하는 확장 매개변수(i번째 레이어의 경우 2 i)가 주어집니다. 그런 다음 이 인과 네트워크의 출력은 시간 차원을 압축하고 모든 시간 정보를 고정 크기 벡터로 집계하는 글로벌 최대 풀링 레이어에 주어집니다(전체 컨볼루션을 사용하는 감독 환경에서 Wang 등(2017)이 제안한 대로). 이 벡터의 선형 변환은 입력 길이, 크기와 무관하게 고정된 인코더의 출력이 됩니다.

# &lt; Experimental Results &gt;

이 섹션에서는 학습된 표현의 관련성을 조사하기 위해 수행한 실험을 검토합니다. 이러한 실험에 해당하는 코드는 보충 자료에 첨부되어 있으며 공개적으로 사용할 수 있습니다.4 전체 학습 과정과 하이퍼파라미터 선택은 보충 자료의 섹션 S1 및 S2에 자세히 설명되어 있습니다. 구현에는 Python 3을 사용했으며, 신경망에는 PyTorch 0.4.1(Paszke et al., 2017)을, SVM에는 scikit-learn(Pedregosa et al., 2011)을 사용했습니다. 각 인코더는 달리 명시되지 않는 한, CUDA 9.0이 탑재된 단일 엔비디아 타이탄 Xp GPU에서 Adam 옵티마이저(Kingma & Ba, 2015)를 사용하여 훈련되었습니다. Selecting hyperparameters for an unsupervised method is challenging since the plurality of downstream tasks is usually supervised. 따라서 Wu 등(2018a)과 마찬가지로 고려되는 각 데이터 세트 아카이브에 대해 다운스트림 작업에 관계없이 단일 하이퍼파라미터 세트를 선택합니다. 또한, 우리는 TimeNet과 같은 다른 비지도 작업과 달리 어떤 작업에 대해서도 비지도 인코더 아키텍처 및 훈련 파라미터의 하이퍼파라미터 최적화를 수행하지 않는다는 점을 강조합니다(Malhotra et al., 2017). 특히 분류 작업의 경우 인코더 훈련 중에 라벨을 사용하지 않았습니다.


<div style="font-size: 20px; margin: 5px 0;"><strong>Classification</strong><br></div>
먼저 시계열 분류에 사용하여 표준 방식(Xu et al., 2003; Dosovitskiy et al., 2014)으로 지도 작업에 대한 학습된 표현의 품질을 평가합니다. 이 환경에서

we show that 

(1) our method outperforms state-of-the-art unsupervised methods, and notably achieves performance close to the supervised state of the art, 

(2) strongly outperforms supervised deep learning methods when data is only sparsely labeled, 

(3) produces transferable representations.

그런 다음 데이터 세트의 훈련 레이블을 사용하여 radial basis function kernel on top of the learned features을 사용하여 SVM을 훈련하고 테스트 세트에서 해당 분류 점수를 출력합니다. 우리의 훈련 절차는 서로 다른 시계열의 표현을 분리할 수 있도록 권장하므로, 이러한 특징에 대한 간단한 SVM의 분류 성능을 관찰하면 품질을 확인할 수 있습니다(Wu et al., 2018a). 또한 SVM을 사용하면 인코더를 훈련할 때 시간(대부분의 경우 몇 분이면 훈련이 완료됨)과 공간 측면에서 모두 효율적인 훈련이 가능합니다.

K가 성능에 큰 영향을 미치므로, 서로 다른 값의 K로 훈련된 인코더로 계산된 표현을 연결하여(자세한 내용은 섹션 S2 참조) 결합된 버전의 방법을 제시합니다. 이를 통해 서로 다른 파라미터로 학습된 표현을 서로 보완하고 분류 점수에서 노이즈를 제거할 수 있습니다.

<div style="font-size: 20px; margin: 5px 0;"><strong>Univariate Time Series</strong><br></div>
다양한 단변량 데이터 세트의 표준 세트인 새로운 UCR 아카이브(Dau et al., 2018)의 128개 데이터 세트 전체에 대한 정확도 점수를 제시합니다. 표 1에는 일부 UCR 데이터 세트에 대한 점수만 보고하고, 모든 데이터 세트에 대한 점수는 보충 자료인 섹션 S3에 보고합니다. 먼저, 학습된 표현을 기반으로 간단한 분류기를 훈련하고 몇 가지 UCR 데이터 세트에 대한 결과를 보고하는 두 가지 비지도 방법인 TimeNet(Malhotra et al., 2017)과 RWS(Wu et al., 2018a)와 이 작업의 두 가지 동시 방법과 우리의 점수를 비교합니다. 또한 아카이브5의 첫 번째 85개 데이터 세트에 대해 Bagnall 외.(2017)이 연구한 지도형 최첨단 분류기 중 최고의 분류기 4개를 비교합니다: COTE(개선된 버전인 HIVE-COTE로 대체됨)(Lines et al., 2018), ST(Bostrom & Bagnall, 2015), BOSS(Schäfer, 2015), EE(Lines & Bagnall, 2015) 등이 있습니다. HIVE-COTE는 계층적 투표 구조에서 많은 분류자를 사용하는 강력한 앙상블 방법이고, EE는 더 간단한 앙상블 방법이며, ST는 셰이프렛을 기반으로 하고, BOSS는 사전 기반 분류기입니다.6 또한 DTW(DTW를 측정값으로 하는 가장 가까운 이웃 분류기)를 기본으로 추가합니다. HIVE-COTE는 앙상블에 ST, BOSS, EE 및 DTW를 포함하므로 이들보다 성능이 우수할 것으로 예상됩니다. 또한, 이스마일 파와즈 외(2019)의 리뷰에서 연구된 최고의 지도 신경망 방법인 Wang 외(2017)의 ResNet 방법과 비교합니다.

![image](https://github.com/passion3659/passion3659.github.io/assets/89252263/95fd382d-08bb-4a13-b696-c552381e733c)


<div style="font-size: 20px; margin: 5px 0;"><strong>performance</strong><br></div>
성능. 비지도 최신 기술(보충 자료의 섹션 S3, 표 S3)과 비교한 결과, 우리의 방법은 비지도 방법인 TimeNet과 RWS(12개 중 11개, 11개 중 10개 UCR 데이터 세트에서)와 일관되게 일치하거나 더 나은 성능을 보여줌으로써 그 성능을 입증했습니다. 저희의 작업과 달리, UCR 아카이브에는 이러한 방법에 대한 코드와 전체 결과가 제공되지 않으므로 결과가 불완전합니다. 지도를 받는 비지도 신경망의 최신 기술과 비교할 때(보충 자료의 그림 S2 및 S3 참조), 우리의 방법은 전 세계적으로 두 번째로 우수하며(평균 순위 2.92), HIVE-COTE(1.71)에 이어 ST(2.95)와 동등하다는 것을 관찰할 수 있습니다. 따라서 우리의 비지도 방식은 여러 유명한 지도 분류기보다 우수하며, 강력한 앙상블 방식보다 앞서는데, 이는 후자가 수많은 분류기와 데이터 표현을 활용하기 때문에 예상했던 결과입니다. 또한, 그림 3은 최대 달성 정확도 대비 정확도 비율의 중앙값이 HIVE-COTE에 이어 두 번째로 우수하고 ST보다 높다는 것을 보여줍니다. 마지막으로, 완전 지도 신경망에 대한 Ismail Fawaz 등(2019)의 연구 결과(보충 자료의 섹션 S3, 표 S3)에 따르면 71개의 UCR 데이터 세트 중 63%에서 우리의 방법을 능가하는 것으로 나타났습니다.7 전반적으로 우리의 방법은 최고의 지도 신경망에 근접하고 두 번째로 잘 연구된 비신경망 지도 방법과 일치하며 특히 HIVE-COTE에 포함된 최고 성능의 방법 수준이기 때문에 놀라운 성능을 달성합니다

<div style="font-size: 20px; margin: 5px 0;"><strong>multivariate time series</strong><br></div>
단변량 시계열만 포함된 UCR 아카이브에 대한 평가를 보완하기 위해 다변량 시계열에 대한 방법을 평가합니다. 이는 제안된 인코더의 첫 번째 컨볼루션 계층의 입력 필터 수를 변경하는 것만으로도 가능합니다. 새로 출시된 UEA 아카이브의 30개 데이터 세트 모두에서 우리의 방법을 테스트합니다(Bagnall et al., 2018). 전체 정확도 점수는 보충 자료 섹션 S4, 표 S5에 나와 있습니다. UEA 아카이브는 단변량 계열에 대한 UCR 아카이브와 같이 다변량 시계열 분류를 위한 표준 아카이브를 제공하기 위한 첫 번째 시도로 설계되었습니다. 출시된 지 얼마 되지 않았기 때문에 다변량 시계열을 위한 최신 분류기들과 비교할 수는 없습니다. 하지만, Bagnall 외(2018)의 연구 결과를 기준으로 DTWD와 비교를 해보았습니다. DTWD(차원 의존적 DTW)는 다변량 환경에서 DTW를 확장할 수 있는 방법이며, Bagnall 등(2018)이 연구한 가장 좋은 기준선입니다. 전반적으로, 우리의 방법은 UEA 데이터 세트의 69%에서 DTWD와 일치하거나 더 나은 성능을 보였으며, 이는 좋은 성능을 나타냅니다. 이 아카이브는 앞으로 계속 성장하고 발전할 예정이므로 추가 비교 없이는 추가적인 결론을 내릴 수 없습니다.

<div style="font-size: 20px; margin: 5px 0;"><strong>Evaluation on Long Time Series</strong><br></div>
회귀 작업에 대한 라벨링이 없는 긴 시계열에 대한 방법의 적용 가능성과 확장성을 보여줌으로써 산업 응용 분야에 해당할 수 있으며, 데이터 세트가 대부분 짧은 시계열을 포함하는 UCR 및 UEA 아카이브에서 수행한 테스트를 보완할 수 있습니다

# &lt; conclusion &gt;

확장 가능하고 고품질의 사용하기 쉬운 임베딩을 생성하는 시계열에 대한 비지도 표현 학습 방법을 제시합니다. 이 방법은 가변 길이 입력을 허용하는 확장 컨볼루션으로 구성된 인코더로 생성되며, 시계열에 대해 새로운 시간 기반 네거티브 샘플링을 사용하여 효율적인 삼중 손실로 훈련됩니다. 실험을 통해 이러한 표현은 보편적이며 최첨단 성능을 달성하는 분류 및 회귀와 같은 다양한 작업에 쉽고 효율적으로 사용할 수 있음을 보여주었습니다.

<div style="font-size: 20px; margin: 5px 0;"><strong>참고 문헌 및 출처</strong><br></div>
<a href="http://dsba.korea.ac.kr/seminar/?mod=document&uid=1817">http://dsba.korea.ac.kr/seminar/?mod=document&uid=1817</a>

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
