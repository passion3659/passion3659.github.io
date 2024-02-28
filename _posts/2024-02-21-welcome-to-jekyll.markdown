---
layout: post
title:  "Pointnet review"
date:   2024-02-21 16:21:44 +0900
categories: pointnet, 3d 
# image:
#     path: /images/figure.jpg
#     thumbnail: /images/pointnet++/화이트.jpg
#     caption: /images/pointnet++/그레이.jpg
---
# PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space

[https://proceedings.neurips.cc/paper_files/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf](https://proceedings.neurips.cc/paper_files/paper/2017/file/d8bf84be3800d12f74d8b05e9b89836f-Paper.pdf)

### Abstract

Pointnet은 metric space points에서 induced된 local structure를 capture하지 못한다. 그래서 복잡한 scene과 세밀한 패턴에 대한 인지능력이 제한된다. Pointnet을 재귀적으로 적용하는 neural network를 도입한다. 점 집합은 일반적으로 다양한 밀도로 샘플링되기때문에 균일한 밀도로 훈련하는 네트워크는 성능이 좋지 않다. 여러 스케일의 특징을 적응하며 결합하는 새로운 집합 학습 계층을 제안한다. 

### 1 Introduction

3d point는 permutation invariant해야하며 여러 local neighborhood의 다른 properties를 나타낼수 있는 metric을 써야한다. 3D는 밀도에 따라 다른 영역을 표시할수 있기 때문이다.

선행 pointnet의 아키텍쳐를 생각해보면 local 특징을 생각하지 않는다 하지만 convoluition 아키텍쳐처럼 로컬 구조를 활용하면 중요하다는 것은 입증되어있다. cnn 구조를 생각해보면 계층구조에 따라 점점 더 큰 규모로 특징을 점진적으로 캡처할 수 있다. 

메트릭 공간에서 샘플링된 포인트 집합을 계층적 방식으로 처리하기 위해 PointNet++이라는 이름의 계층적 신경망을 소개합니다. PointNet++의 일반적인 개념은 간단합니다. 먼저 기본 공간의 거리 메트릭에 따라 포인트 집합을 겹치는 로컬 영역으로 분할합니다. CNN과 유사하게, 작은 영역에서 미세한 기하학적 구조를 포착하는 로컬 피처를 추출하고, 이러한 로컬 피처를 더 큰 단위로 그룹화하여 더 높은 수준의 피처를 생성하도록 처리합니다. 이 과정은 전체 포인트 세트의 특징을 얻을 때까지 반복됩니다.

The design of PointNet++ has to address two issues

1. how to generate the partitioning of the point set
2. how to abstract sets of points or local features through a local feature learner

근데 partitioning된 point들이 같은 구조를 생산해야하고 local feature learners의 가중치가 shared되야하기때문 convolution과 매우 비슷하다. 이 논문에서는 **local feature learner를 pointnet으로 선택했다**. pointnet은 정렬되지 않은 데이터 포인트 집합을 처리하는데 효과적인 아키텍처이다. 또한 손상에도 강하다. PointNet++는 입력 세트의 중첩된 파티셔닝에 PointNet을 재귀적으로 적용한다. 

여전히 남아 있는 한 가지 문제는 점 집합의 겹치는 분할을 생성하는 방법입니다. 각 파티션은 기본 유클리드 공간의 이웃 공으로 정의되며, 매개변수에는 중심 위치와 배율이 포함됩니다. 전체 집합을 균등하게 커버하기 위해 가장 먼 지점 샘플링(FPS) 알고리즘에 의해 설정된 입력 지점 중에서 중심점이 선택됩니다. 고정된 보폭으로 공간을 스캔하는 볼류메트릭 CNN에 비해, 로컬 수신 필드는 입력 데이터와 메트릭에 따라 달라지므로 더 효율적이고 효과적입니다.

그러나 특징 스케일의 얽힘과 입력 포인트 세트의 불균일성으로 인해 국부적인 이웃 볼의 적절한 스케일을 결정하는 것은 더 어렵지만 흥미로운 문제입니다. 우리는 입력 포인트 세트가 서로 다른 영역에서 가변 밀도를 가질 수 있다고 가정하며, 이는 구조 센서 스캐닝 [18]과 같은 실제 데이터에서 매우 일반적입니다(그림 1 참조). 따라서 우리의 입력 포인트 세트는 균일하고 일정한 밀도를 가진 일반 그리드에 정의된 데이터로 볼 수 있는 CNN 입력과는 매우 다릅니다. CNN에서 로컬 파티션 스케일에 대응하는 것은 커널의 크기입니다. [25]는 더 작은 커널을 사용하면 CNN의 성능을 향상시키는 데 도움이 된다는 것을 보여줍니다. 그러나 포인트 집합 데이터에 대한 우리의 실험은 이 규칙에 반하는 증거를 제시합니다. 작은 이웃은 샘플링 부족으로 인해 너무 적은 수의 점으로 구성될 수 있으며, 이는 포인트넷이 패턴을 강력하게 포착하기에 불충분할 수 있습니다.

이 백서의 중요한 기여는 PointNet++가 여러 스케일의 이웃을 활용하여 견고성과 디테일 캡처를 모두 달성한다는 것입니다. 훈련 중 무작위 입력 드롭아웃을 통해 네트워크는 다양한 스케일에서 감지된 패턴에 적응적으로 가중치를 부여하고 입력 데이터에 따라 다중 스케일 특징을 결합하는 방법을 학습합니다. 실험 결과 PointNet++은 포인트 세트를 효율적이고 강력하게 처리할 수 있는 것으로 나타났습니다. 특히 3D 포인트 클라우드의 까다로운 벤치마크에서 최첨단 기술보다 훨씬 우수한 결과를 얻었습니다.

### 2. problem statement

semantic interest가 적용된 x를 뽑아내는 것이다. 그래서 그 x를 classification과 segmentation에 적용했을때 성능이 좋게 나오는 것이다.

### 3. Method

**3.1 review of pointnet**

이건 했으니까 됐음

**3.2 Hierarchical Point Set Feature Learning**

pointnet은 max pooling으로 전체 포인트 집합을 집계했지만 우리가 제안하는 새로운 아키텍쳐는 계층적 포인트 그룹을 구축하고 계층을 따라 점점 더 큰 로컬 영역으로 계산하는것이다. set abstraction level은 three key layer로 이루어져 있다. 

**Sampling layer,     Grouping layer,    PointNet  layer**

**sampling layer**

Sampling layer는 포인트 집합을 선택해서 로컬 영역의 중심을 정의한다.

그룹화레이어는 인접한 점을 찾아 로컬 영역의 집합으로 구성한다. 

pointnet layer는 로컬 역역 패턴을 특징 벡터로 인코딩한다. 

![image](https://github.com/passion3659/passion3659.github.io/assets/89252263/5e2bf7a4-9981-47dd-95d8-eca96e0487a0)

**grouping layer**

**pointnet layer**

**3.3 Robust Feature Learning under Non-Uniform Sampling Density**

앞서 설명한 것처럼, 점 집합은 영역마다 밀도가 균일하지 않은 것이 일반적입니다. 이러한 불균일성은 포인트 집합 특징 학습에 상당한 문제를 야기합니다. 밀도가 높은 데이터에서 학습된 특징이 드물게 샘플링된 영역으로 일반화되지 않을 수 있습니다. 따라서 희박한 포인트 클라우드에 대해 학습된 모델은 세분화된 로컬 구조를 인식하지 못할 수 있습니다.

이상적으로는 샘플링이 밀집된 영역에서 세밀한 디테일을 포착하기 위해 설정한 지점을 최대한 면밀히 검사하는 것이 좋습니다. 그러나 밀도가 낮은 지역에서는 샘플링 결핍으로 인해 로컬 패턴이 손상될 수 있으므로 이러한 정밀 검사는 금지되어 있습니다. 이 경우 더 넓은 범위에서 더 큰 규모의 패턴을 찾아야 합니다. 이 목표를 달성하기 위해 우리는 입력 샘플링 밀도가 변할 때 서로 다른 스케일의 영역에서 특징을 결합하는 방법을 학습하는 밀도 적응형 포인트넷 레이어를 제안합니다(그림 3). 밀도 적응형 포인트넷 레이어가 포함된 계층적 네트워크를 포인트넷++라고 부릅니다.

이전 섹션 3.2에서 각 추상화 수준에는 단일 스케일의 그룹화 및 특징 추출이 포함되어 있었습니다. PointNet++에서 각 추상화 수준은 로컬 패턴의 여러 스케일을 추출하고 로컬 포인트 밀도에 따라 지능적으로 결합합니다. 로컬 영역을 그룹화하고 서로 다른 스케일의 특징을 결합하는 측면에서 아래 나열된 두 가지 유형의 밀도 적응형 레이어를 제안합니다.

Multi-scale grouping (MSG).

그림 3 (a)에서 볼 수 있듯이, 멀티스케일 패턴을 캡처하는 간단하지만 효과적인 방법은 스케일이 다른 그룹화 레이어를 적용한 다음 포인트넷에 따라 각 스케일의 특징을 추출하는 것입니다. 서로 다른 스케일의 특징을 연결하여 멀티스케일 특징을 형성합니다. 멀티 스케일 특징을 결합하는 최적화된 전략을 학습하도록 네트워크를 훈련시킵니다. 이는 각 인스턴스에 대해 무작위 확률로 입력 포인트를 무작위로 탈락시키는 방식으로 이루어지며, 이를 무작위 입력 탈락이라고 합니다. 구체적으로, 각 훈련 포인트 세트에 대해 [0, p]에서 균일하게 샘플링된 드롭아웃 비율 θ를 선택합니다(여기서 p ≤ 1). 각 점에 대해 확률 θ로 무작위로 점을 삭제합니다. 실제로는 빈 점 집합을 생성하지 않기 위해 p = 0.95로 설정합니다. 이렇게 함으로써 네트워크에 다양한 희소성(θ에 의해 유도됨)과 다양한 균일성(드롭아웃의 무작위성에 의해 유도됨)을 가진 훈련 집합을 제공합니다. 테스트 중에는 사용 가능한 모든 포인트를 유지합니다.

Multi-resolution grouping (MRG)

위의 MSG 접근 방식은 모든 중심점에 대해 대규모 이웃에서 로컬 포인트넷을 실행하기 때문에 계산 비용이 많이 듭니다. 특히 최저 수준에서 중심점의 수가 일반적으로 상당히 많기 때문에 시간 비용이 상당히 많이 듭니다. 여기서는 이러한 고비용 계산을 피하면서도 포인트의 분포 특성에 따라 정보를 적응적으로 집계하는 기능을 유지하는 대안을 제안합니다. 그림 3 (b)에서 특정 레벨의 영역 특징 Li는 두 벡터의 연결입니다. 하나의 벡터(그림 왼쪽)는 설정된 추상화 수준을 사용하여 하위 레벨 Li-1에서 각 하위 영역의 특징을 합산하여 얻습니다. 다른 벡터(오른쪽)는 단일 포인트넷을 사용하여 로컬 영역의 모든 원시 포인트를 직접 처리하여 얻은 특징입니다. 로컬 영역의 밀도가 낮으면 첫 번째 벡터를 계산할 때 하위 영역에 더 희박한 점이 포함되어 있고 샘플링 결핍이 더 심하기 때문에 첫 번째 벡터가 두 번째 벡터보다 신뢰도가 떨어질 수 있습니다. 이러한 경우 두 번째 벡터에 더 높은 가중치를 부여해야 합니다. 반면, 로컬 영역의 밀도가 높은 경우 첫 번째 벡터는 낮은 레벨에서 더 높은 해상도로 재귀적으로 검사할 수 있는 기능이 있기 때문에 더 세밀한 정보를 제공합니다. 이 방법은 가장 낮은 레벨에서 대규모 이웃의 특징 추출을 피하기 때문에 MSG에 비해 계산적으로 더 효율적입니다.

3.4 Point Feature Propagation for Set Segmentation

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
