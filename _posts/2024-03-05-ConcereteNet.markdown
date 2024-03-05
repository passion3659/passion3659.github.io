---
layout: post
title:  "ConcreteNet review"
date:   2024-03-5 16:16:23 +0900
use_math: true
categories:
  - NLP
  - 3D
  - Multimodal
  - Paper Review
---
# "Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding"

원문: <a href="https://arxiv.org/abs/2309.04561">Unal, Ozan, et al. "Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding." arXiv preprint arXiv:2309.04561 (2023).</a> 

{% include toc %}

제가 연구하려는 분야의 가장 성능이 높은 모델의 논문인데 재미있게 읽어보겠습니다.

# &lt; Abstract &gt;

3D visual grounding는 자연어로 된 설명으로 참조되는 3D 장면에서 물체의 위치를 파악하는 작업 보통 bbox(bounding box)를  통해서 진행한다.하지만 실제 application에서는 bbox는 충분히 설명하지 못한다. 따라서 3D-visual grounding보다는 3D instance segmentation로 진행하려고 한다.

본 논문은 세가지 새로운 독립형 모듈을 통해 예측에  방해되는 동일한 클래스의 여러 instance의 정보를 최소화시켜 성능을 개선하는 ConcreteNet을 제안한다. 

ConcreteNet의 구조는 다음과 같다.

1. 먼저, 인스턴스 간 관계 단서를 모호하게 만드는 bottom-up attentive fusion module을 거친다
2. 다음으로 latent space에서 분리를 유도하는 contrastive training 체계를 구축힌디, 
3. 마지막으로 Learned global camera token을 통해 view-dependent utterances를 해결한다. 

# &lt; Introduction &gt;

2D에 비해  3D point cloud의 멀티 모달리티는 매우 복잡하고 어려운 작업으로 남아 있다. 최근 3D visual grounding에 대한 관심이 높아진 것은 초석 데이터 세트인 ScanRefer, Nr3D 및 Sr3D의 도입 덕분이다. 

Scanrefer와 Nr3d, Sr3D는 input으로 받는 description에서 차이가 있는데 그 때문에 task가 서로 다르다. Scarefer는 point cloud와 text를 받아서 bbox를 생성하는 것이라면 Nr3d와 Sr3D는 point cloud와 text를 받고 또 같은 class에 속하는 후보 bbox를 input으로 받아서 식별합니다. 그래서 Scanrefer로 하는 3D visual Grounding이 더 어려운 task가 된다. Concretenet은 Scanrefer의 task에 더 중점을 둔다. 

Abstract에서 말했던거 처럼 bbox는 실제 application에 사용이 제한적이다. 예를 들어 로봇이 물체를 잡거나 피해야하면 bbox보다는 점들을 다 인식하는것이 좋다. 아래 그림을 보면 ConcreteNet에서 나온 segmantation 정보가 더 정확한 것을 볼 수 있다.

![image](https://github.com/passion3659/2023_Crime_Safety_Data_Analysis_Competition/assets/89252263/c4404442-5b6d-4c97-8b52-8ba5eb20e63d)

grounding-by-selection strategy을 이용한 새로운 모델인 ConcreteNet의 단계는 다음과 같다. 

1. grounding-by-selection에서는 먼저 시각적 백본이 포인트 클라우드를 입력으로 하여 3D 인스턴스 후보를 생성한다. 
2. 그 후, verbo-visual fusion module이 자연어 설명을 기반으로 올바른 후보를 선택한다. 

그런데 이 방식은 같은 종류의 객체들 사이에서는 이러한 방식이 잠재 공간에서 객체들을 충분히 구분하지 못하는 문제를 가지고 있다는 점도 문제점이 된다. 

ex) 쉽게 설명하자면 여러 개의 의자가 포함된 3D 환경을 생각해보자. 각 의자는 개별적으로 구분되어야 하지만, 모두 '의자'라는 동일한 카테고리에 속한다. ConcreteNet 같은 시스템이 이러한 의자들을 개별 객체로 인식하고 각각을 정확히 분리해내려고 할 때, 모든 의자가 비슷한 "의자"라는 추상적 특징을 공유하기 때문에, 이들을 개별적인 인스턴스로 구분하는 것이 어려울 수 있다. 즉, 잠재 공간에서 이러한 의자들의 표현이 서로 가까워지면서, 모델이 그들 사이를 명확히 구분하는 데 어려움을 겪는다는 것이다. 

이 문제를 해결하기 위해 dense 3D visual grounding를 위해 verbo-visual fusion을 개선하는 세 가지 새로운 방법을 제안한다. 

1. 첫번째로 여러 참조에 대해 locality rules이 적용된다는 것을 관찰해 Bottom-up Attentive Fusion Module (BAF)을 이용해서 성능을 항상시킨다.
    1. locality rules은 가리키는 대상을 찾을 때, 가까운 대상부터 차례대로 확인하는 방식을 의미하는데 이 방식이 모델에 적용되면 더 정확도가 올라간다.
2. 두번째로 같은 class에 대해서 설명이 해당 객체이면 가깝게 다른 객체이면 멀리 훈련하는 contrastive training을 적용한다.
    1. 동일한 의미론적 클래스에 속하는 반복적 인스턴스 간의 모호성 때문에 latent verbo-visual space 내에서 인스턴스를 분리하는 것이 어렵기 때문에 적용한다. 
3. 세번째로 기계가 발화와 관련된 가능한 시점을 이해하고 공감할 수 있도록, 'Learned Global Camera Token (GCT)'을 도입한다.
    1. 3D 장면은 2D와 달리 본질적으로 방향성이 있는 오른쪽이나 왼쪽, 뒤나 앞을 갖지 않는다. 그러나 실제 상황에서 인식은 개인적인 관점에 의해 안내되므로, 시점 의존적 설명은 피할 수 없다.

본 논문에서 contribution은 위의 세가지 접근법에 대해서라고 생각하면 된다. 

# &lt; Related Work &gt;

<div style="font-size: 20px; margin: 5px 0;"><strong>3D visual grounding</strong><br></div>

3D visual grounding은 이미지에 대한 언어적 설명을 해당 이미지의 특정 객체에 매핑하는 2D 시각적 정합의 3D 버전이다. 포인트 클라우드 형태의 3D 장면을 입력으로 사용하며, 설명을 3D에서 언급된 객체에 찾는 것을 목표로 한다. 

3D-language tasks : 3D dense captioning [14], [15] 그리고 grounding spatial relations (instead of individual objects) in 3D [16]가 있다. 

주목할 만한 초기 작업 : NYU Depth-v2 데이터셋을 사용하여 3D 장면과 그 설명 사이의 관계를 공동으로 추론하는 방법을 탐구했다.

object proposal 모듈과 fusion 모듈에서 attetion mecahinism을 이용해 시도[7]되었는데 본 논문은 이를 활용해 개선한 융합 방식을 제안한다. 개선 방식은 위에 있는 3개의 contribution이다.

그리고 전의 연구와 접근방식을 비교해서 정리하면 다음과 같다. 

- global attention과 local attetion
    - 이전 연구: 트랜스포머 디코더 아키텍처를 사용하여 객체 간 정보 라우팅에global attention를 의존하는 fusion을 구현한다.
    - 본 논문 접근 방식: 객체 간 공간적 관계를 더 명확하게 구별하기 위해 spherical attention masks 방식을 통해 지역성을 명시적으로 유도한다.
- local attetion의 구현:
    - [8]: 3D 장면을 거친 보크셀로 나누고 각 보크셀 내의 다양한 시각적 임베딩 간 주의를 제한하여 local attetion를 구현한다.
    - 본 논문 접근 방식: spherical attention masks를 사용하여 객체의 중심에서 등방성 방식으로 지역성을 구현한다.
- 시점 의존성:
    - MVT-3DVG [19]: 입력 3D 장면에 회전 증강을 적용하여 데이터 기반 방식으로 시각 기반 객체 임베딩의 시점 민감성을 해결한다.
    - 본 논문 접근 방식: 그라운딩 모델의 이미 큰 3D 입력을 늘리는 대신, 시각 토큰에 추가적인 카메라 시점 토큰을 포함시켜 fusion에 시점 민감성을 주입한다.
- 시점 정보의 효과:
    - [20]의 실증적 발견: Nr3D 장면에서 시점 의존적 설명을 가진 3D 객체 식별 정확도에 대략적인 시점 정보가 성능을 올려준다.
    - 본 논문 접근 방식: ScanRefer 데이터의 정확한 주석을 통해 시점을 학습하고, 이 데이터에서 풍부한 시점 의존적 설명을 활용한다.

<div style="font-size: 20px; margin: 5px 0;"><strong>Dense 3D visual grounding</strong><br></div>
이건 sementic segementation이라고 보면 되는데 당연히 3D는 훨씬 덜 탐구되었다. 기존 연구와 본 논문 접근 방식에 대해서 보겠다.

- 기존 연구
    - [9] : 인스턴스 임베딩과 단어 임베딩 간의 fusion을 그래프 신경망을 사용하여 수행한다.
    - [10] : 전역 문장 임베딩만을 입력으로 받아, 인스턴스 임베딩이 기하학적 및 외형적 특징에 대한 구체적인 정보를 가진 개별 단어에 주의를 기울일 수 없게 한다.
- 본 논문 접근 방식
    - 인스턴스 후보 추출 과정에서 생산된 semantic 인스턴스 특성은 [10]에서는 인스턴스 임베딩 추출을 위한 후속 과정에서 버려진다. 반면에, 본 논문은 이러한 의미론적 특성을 처음부터 끝까지(end-to-end) 학습하여, 정합과 분할 정확도에 대해 구별력 있는 인스턴스 임베딩 생성을 위한 최적화를 모두 수행한다.

# &lt; METHOD &gt;

세 가지 핵심 구성 요소로 이루어져 있다. 1) 3D 인스턴스 후보를 생성하는 3D 시각적 백본과 그에 따른 마스크를 소개한다. 2) 언어 쿼리를 고차원 단어 임베딩으로 인코딩하는 방법을 설명한다. 3) 참조된 인스턴스 마스크를 예측하기 위해 단어 임베딩과 인스턴스 임베딩을 융합하여 3D 공간에서 설명을 근거로 하는 fusion 모듈을 제시한다. 전체 파이프라인은 다음 그림과 같다. 

![image](https://github.com/passion3659/2023_Crime_Safety_Data_Analysis_Competition/assets/89252263/09864065-2980-4098-8d09-4303f9ac8115)

포인트 클라우드와 자연어 프롬프트가 주어지면 먼저 인스턴스 후보(파란색)와 단어 임베딩(분홍색)을 생성한다. 그런 다음 이를 융합하여 언어적 설명을 3D 장면에 조밀하게 fuse한다. bottom-up attentive fusion module(BAF)을 통해 attention을 localize하고, contrast learning을 활용하여 특징의 분리 가능성을 높이고, 카메라 위치를 학습하여 뷰에 따라 달라지는 설명을 모호하게 함으로써 성능을 개선합니다. 최종 예측은 가장 잘 맞는 인스턴스의 토큰을 예측된 마스크와 병합하여 생성된다. 

## **Kernel-Based 3D Instance Segmentation**

<div style="font-size: 20px; margin: 5px 0;"><strong>feature extraction</strong><br></div>
그림과 같이 UNet을 backbone으로 사용해서 point를 처리하는 구조를 가진다. 이때 두개의 loss를 이용하는데 sementic loss와 offset loss이다. semantic loss는 UNet에서 나온 값과 실제 semantic label을 비교하는 loss이고 offset loss는 instance center에서의 거리를 ground truth와 pred value와 비교하는 값이다. 수식은 아래와 같다. 

$$
\begin{equation*}
L_{sem} = H(s, \hat{s})
\end{equation*}
$$

$$
\begin{equation*}
L_{off} = L_1(x, \hat{x})
\end{equation*}
$$

<div style="font-size: 20px; margin: 5px 0;"><strong>Candidate generation</strong><br></div>
Unet을 거쳐서 나온feature들을 MLP를 통과시켜서 centroid map h를 만든다. centorid map은 아래의 loss에 의해 supervised되어서 만들어진다. p는 indicator함수로써 instnace라고 속할때 1의 값을 가진다. 

$$
\begin{equation*}
L_{cen} = \frac{1}{\sum_{i=1}^{N} 1(p_i)} \sum_{i=1}^{N} | h_i - \exp \left( -\frac{\gamma \| x_i \|^2}{b_i^2} \right) |
\end{equation*}
$$

예측된 히트맵에서 local normalized non-maximum suppression (LN-NMS)을 통해 후보를 생성합니다. 이때의 loss는 다음과 같고 a hat이 ground truth인 centorid map이며 다음 loss에 의해 supervised 학습된다.

$$
\begin{equation*}
L_{agg} = L_1(a, \hat{a})
\end{equation*}
$$

 그렇게 해서 나오는 후보들의 최종 loss는 위 loss의 합으로 다음과 같다. 

$$
\begin{equation*}
L_{can} = L_{sem} + L_{off} + L_{cen} + L_{agg}
\end{equation*}
$$

<div style="font-size: 20px; margin: 5px 0;"><strong>mask generation</strong><br></div>
마스크는 segmentation에서 중요한 역할을 하는데 후보들이 객체가 아니라고 판단되면 0으로 loss 계산에서 제외시켜줘야하기 때문이다. 이 mask도 threshold에 의해 결정되는데 본 논문에서는 IoUrk 0.25가 넘으면 객체라고 판단한다. 그럼 이 threshold에 의해서 생긴 0과 1도 label이 되어서 학습이 되는데 DyCo3d 방법을 따라 동적 컨볼루션을 사용하여 최종 인스턴스 마스크와 계산이 되며 loss는 다음과 같다. 

$$
\begin{equation*}
L_{mask} = H_b(z, \hat{z}) + DICE(z, \hat{z})
\end{equation*}
$$

## Encoding Language Cues

자연어 처리에서 본 task는 초기에 Glove를 통과시켜 GRU를 거쳐서 나온 feature를 concat했는데 최근에는 엄청난 자연어의 발전에 의해서 이 방법또한 구식이 되었다. 본 논문에서는 MPNet을 거쳐서 나온 값을 single linear layer를 통과시킨 값을 Language의 feature로 사용한다. MPNet은 리뷰한 사람이 꽤 있어서 구글링하면 잘 나온다. 아래는 MPNet의 구조이다.

![image](https://github.com/passion3659/2023_Crime_Safety_Data_Analysis_Competition/assets/89252263/5be93478-bd14-45e4-a460-9f978457b74d)

## Verbo-visual Fusion

일반적으로는 후보들이 있으면 feature를 fusion해서 나온 확률값으로 선택을 하는 메커니즘을 거치는데 이 fusion은 보통 MLP로 사용하였다. 하지만 최근에는 트랜스포머 아키텍쳐를 사용해 fuse를 보통 한다. 다음 공식으로 하는데 transformer를 공부하면 공식은 익숙할것이다.(모른다면 “구글 bert의 정석” 책으로 공부 강추!) 

$$
\begin{equation*}
f_l = \text{softmax}(qk_l^T)v_l + f_{l-1}
\end{equation*}
$$

그러나 instance segmentation은 3D 실내 측위에서 3D object detection보다 낫지는 않더라도 동등한 성능을 보이지만, 고밀도 커널 기반 모델은 동일한 semantic class의 인스턴스 간 잠재 공간에서 분리 가능성이 제한적인 것으로 나타난다. 이를 해결하기 위해 (i) 인스턴스 간 관계 단서를 명확히 하고, (ii) 잠재적 표현에서 더 나은 분리 가능성을 유도하기 위한 훈련을 지원하며, (iii) 센서 위치를 추론하여 뷰 의존적 설명을 해결하는 세 가지 모듈을 제안한다.

<div style="font-size: 20px; margin: 5px 0;"><strong>Bottom-up attentive fusion (BAF)</strong><br></div>
자연어 프롬포트를 생각할 때 예를들어 “캐비닛 옆에 있는 의자”이면 의자라는 객체를 찾을 때 캐비닛의 정보를 이용해 찾는것이라 local한 정보로 찾는다고 볼 수 있다. 그래서 Bottom-up attentive fusion 방식을 이용한다고 본 논문은 제안하는데 구조는 다음 그림과 같다. 

![image](https://github.com/passion3659/2023_Crime_Safety_Data_Analysis_Competition/assets/89252263/5a0129c0-bef4-4670-b0f9-cc82b6a643f9)

BAF(Bottom-Up Attentive Fusion module)에서 트랜스포머 인코더 블록을 사용하는데 이때 localized self-attention로 모델링한다. 다음 공식을 따르는데 

$$
\begin{equation*}
f_l = \text{softmax}(M_l + qk_l^T)v_l + f_{l-1}
\end{equation*}
$$

$$
\begin{equation*}
M_l(i, j) = \begin{cases} 0, & \text{if } ||o_i - o_j|| < r_l \\ -\infty, & \text{otherwise} \end{cases}
\end{equation*}
$$

$r_l$는 거리인데 두개의 후보 $r_l$사이  거리가 $r_l$보다 작으면 마스크 $M_l(i,j)$는 0을 취하며 attention이 계산되고 그렇지않으면 음의 무한대를 가져서 attention 계산에서 제외된다. 그리고 cross-attention과 feed-forward layer를 통해서 instance token과 word embedding을 융합한다.  종 인스턴스 임베딩을 생성하기 위해, 트랜스포머 디코더 블록을 여러 번 반복 적용하며, 각 반복마다 마스킹 반경 $r_l$을 증가시키는 방식으로 작동한다. mask도 supervised로 선택되는데 이때 loss는 cross entropy loss이다.

<div style="font-size: 20px; margin: 5px 0;"><strong>Inducing separation via contrastive learning</strong><br></div>
$$
\begin{equation*}
L_{\text{con}}(e_s, e_i) = -\log \frac{\exp(d(e_s, e_{i,k+}) / \tau)}{\sum_k \exp(d(e_s, e_{i,k}) / \tau)}
\end{equation*}
$$

$e_s$는 문장 임베딩, $e_i$는 인스턴스 임베딩, $d()$는 코사인 유사도, $\tau$는 온도 매개변수를 나타내는데 이 손실 함수는 word embedding과 instance embedding 사이의 구분하기 위함이다. 문장 embedding과 일치하는 인스턴스 벡터는 긍정적인 샘플로 취급되어 서로 가까워지도록 유도되며, 나머지 인스턴스와의 쌍은 부정적인 샘플로 취급되어 서로 멀어지도록 유도된다. 이 방식은 반복적인 인스턴스 매핑 간의 모호성을 줄이고, 다중 인스턴스 참조 상황에서의 지역화를 돕기 위한 일반적인 솔루션을 제공한다. 

<div style="font-size: 20px; margin: 5px 0;"><strong>Global camera token (GCT)</strong><br></div>
3D의 입장에서 물체를 바라볼 때 우리는 관점에 따라 방향이 달라진다. 그래서 카메라가 보고있는 위치와 방향의 정보를 token으로 넣어서 임베딩을 생성한다. 좀 어려운 말로 시점 의존전 참조를 해결한다고 말할수 있다. 카메라 토큰에 대한 마스킹 값은 모든 인스턴스 *i*에 대해 0으로 설정되어, 모든 레이어에서 카메라 token이 attention계산을 하도록 한다. 카메라 토큰의 출력은 주석 과정에서 사용된 카메라 위치와 비교하여 L2 손실을 통해 supervised된다. 이는 모델이 카메라의 관점을 학습하고 이를 바탕으로 객체를 더 정확히 지역화할 수 있도록 돕는다. 카메라 토큰에 대한 마스킹 값은 아래와 같이 정의한다. 

$$
\begin{equation*}
M_l(i, i_c) = M_l(i_c, i) = 0, \ \forall i.
\end{equation*}
$$

# &lt; Experiments &gt;

800개의 ScanNet 장면[6]에서 11,046개의 오브젝트에 대한 51,583개의 설명을 제공하는 ScanRefer 데이터세트[1]에서 실험을 수행한다. Hyperparameter는 다음과 같다. 

**Implementation details**

1. **3D Unet 백본을 위한 보켈 크기(Voxel Size)**: 2cm
2. **자연어 처리를 위한 MPNet 토크나이저와 사전 훈련된 모델**
3. **단어 임베딩 추출 전 물체명사를 무작위로 마스킹할 확률**: 0.5
4. **MPNet의 출력을 128차원 벡터로 리맵핑하기 위한 선형 레이어**
5. **언어 인코딩을 위한 바닐라 트랜스포머 인코더 레이어 수**: 2
6. **주의 융합을 위한 트랜스포머 디코더 레이어 수**: 6
7. **마스킹 구체 반경 (rl)**: [1.0m, 2.5m, ∞]
8. **인스턴스 간 평균 거리에 대한 대략적 반경**: 2.5m
9. **최종 두 레이어에서 전역 주의를 제공하는 반경**: ∞ (무한대)
10. **γ (감마)**: 25 (DKNet을 따름)
11. **τ (타우, Temperature for Softmax)**: 0.3
12. **배치 크기(Batch Size)**: 4
13. **샘플 구성**: 하나의 장면과 최대 32개의 발화
14. **학습 에포크 수**: 400
15. **옵티마이저**: AdamW
16. **학습률(Learning Rate)**: 3 · 10−4
17. **사용된 하드웨어**: Nvidia RTX 3090 GPU

**Results**

![image](https://github.com/passion3659/2023_Crime_Safety_Data_Analysis_Competition/assets/89252263/f7ff3e35-781b-482e-ba5a-1ed956ee05b0)

전반적으로 성능이 매우 향상했다고 보고한다. 

**ablation study**

task가 데이터셋이 하나밖에 없어서 다른 이것저것 실험을 많이 했는데 BAF, contrast learning, GCT를 하고 안하고의 성능은 다음과 같다. 

![image](https://github.com/passion3659/2023_Crime_Safety_Data_Analysis_Competition/assets/89252263/266fd34a-4f75-408f-bfba-225e9fa4812b)

그리고 language backbone으로 썼던건 다음과 같다. 

![image](https://github.com/passion3659/2023_Crime_Safety_Data_Analysis_Competition/assets/89252263/8433ceb0-b3ec-480b-9c2e-531d35d5ace1)

# &lt; Conclusion &gt;

고밀도 3D 시각적 그라운딩, 즉 참조 기반 3D instance segmentation 문제를 다룬다. 기준 커널 기반의 고밀도 3D 접지 접근법을 확립하고, 세 가지 독립적인 개선안을 제안하여 발생하는 약점을 해결한다. 먼저 bottom-up attentive fusion module을 도입하여 인스턴스 간 관계 단서를 찾고, 다음으로 contrastive loss을 구성하여 일치하지 않는 인스턴스와 문장 임베딩 사이의 잠재 공간 분리를 유도하고, 마지막으로 글로벌 카메라 토큰을 학습하여 view-dependent utterances를 명확히 구분한다. 이 세 가지 새로운 모듈을 결합하여 제안된 ConcreteNet은 널리 사용되는 ScanRefer 온라인 벤치마크에서 sota를 보인다.


