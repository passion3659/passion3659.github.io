---
layout: post
title:  "Understanding GPU version"
date:   2024-03-7 19:59:34 +0900
use_math: true
categories:
  - Programming
---
# 딥러닝 환경 설정: GPU와 CUDA, 우분투 설치의 버전 호환성 이해하기

버젼 확인은 딥러닝에서 매우매우 중요하다. 당연히 버전이 안 맞으면 보통 안 돌아가기 때문이다….
깃허브에 있는 코드를 실행시키려면 거기에 있는 요구사항을 맞춰야하는데 일단 어떻게 pytorch와 cuda 버전이 어떻게 설치되는지 아예 처음부터 이해할 필요가 있다. 

# 설치과정

 보통 gpu를 사면 gpu 하드웨어의 버젼이 정해져있다. (RTX3090..이런느낌으로)

그리고 보통 리눅스 기반의 본체에 gpu를 달아서 쓴다. (우리가 보통 쓰는건 window기반 본체!)

여기에서 본체에 usb 꼽고 우분투 운영체제를 설치한다. ex) Ubuntu 20.04.6

그 다음 바로 그냥 CUDA를 설치해준다. 이때 NVIDA Driver도 같이 설치가 된다. ([https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal](https://developer.nvidia.com/cuda-10.1-download-archive-base?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal) 요런거로 설치)

코드로 할거면 다음 블로그를 따라가면 된다. [https://ropiens.tistory.com/34](https://ropiens.tistory.com/34) 

이렇게 하면 우리가 잘 아는 nvidia-smi로 Driver version과 CUDA Version을 확인할 수 있는 것이다. 

# GPU ⇒ CUDA ⇒ PYTORCH 버젼 체크

내가 가지고 있는 버젼이 RTX3090이다. 

그럼 [https://en.wikipedia.org/wiki/CUDA](https://en.wikipedia.org/wiki/CUDA) 를 봤을 때 

![image](https://github.com/daveredrum/ScanRefer/assets/89252263/4d7d5bb4-0731-4e3f-b9fa-8522ebc8f82c)

RTX3090은 은 8.6에 속하는 것을 볼 수 있다. [https://en.wikipedia.org/wiki/CUDA](https://en.wikipedia.org/wiki/CUDA) 에 아래 그림을 보면 쿠다 11.1부터 12.4까지는 호환이 가능하다고 나와있다. 그럼 난 CUDA 11.1부터 12.4까지 사용이 가능한 것이다.

![image](https://github.com/daveredrum/ScanRefer/assets/89252263/e10231c5-0d2a-4e2e-adc5-1471b14a7e6b)

내가 원하는 버젼을 [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive) 에서 다운을 받고 그 다음 torch를 다운받아주는데 이때 버젼확인은 다음 사이트에서 하면 된다. <br>
[https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)