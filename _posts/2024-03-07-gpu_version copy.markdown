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

# 기본적인 background 지식

우리가 평소에 쓰는 컴퓨터 본체(컴퓨터 껐다 켰다 하는 본체)는 cpu가 달려있다.

이 cpu 본체가 있는데 여기에 gpu를 달아서 쓴다. (우리가 흔히 말하는 컴퓨터 본체가 cpu본체라고 생각하자)

cpu본체를 처음 사용하면 아무것도 없기 때문에 window나 우분투같은 운영체제를 설치해야한다. 

처음에 착각을 했었는데 나는 window용 cpu본체, 우분투용 cpu본체가 따로 있는줄 알았다.
그게 아니라 운영체제를 자유롭게 설치하고 설치하는대로 window, 우분투가 되는거다.

아 그리고 우분투가 리눅스 기반 운영체제라서 우분투를 그냥 리눅스라고도 얘기하니까 알잘딱깔센을 잘하자.

# 딥러닝 환경 세팅 설치과정

딥러닝은 우분투에서 많이 이용하기 때문에 우분투 기반으로 설치하는걸 말하겠다.

본체에 usb 꼽고 우분투 운영체제를 설치한다. ex) Ubuntu 20.04.6

이 우분투 설치하는건 많은 블로그가 정리해놨다. 그걸 보고하자. 

설치가 됐다면 터미널에 들어가서 설치가능한 드라이버 버전을 확인해보자.

```python
ubuntu-drivers devices
```

어떤 CUDA를 설치할지는 모르겠지만 아래 그림과 같이 보통 470을 설치한다. 470이 아래 그림에 따르면 다 호환돼서 그럴거다.
![image](https://github.com/passion3659/passion3659.github.io/assets/89252263/7afe2cf5-1ac8-4ad6-afcc-2884bf4dc3ef)

그러니 아래 코드로 드라이버를 설치하고 재부팅을 하자.

```python
sudo apt install nvidia-driver-470
sudo reboot
```

재부팅을하고 드라이버 정보를 확인하면 된다. 이 코드는 너무 유명하다...

```python
nvidia-smi
```
CUDA 버젼이 나와있는데 위에있는 그림에서 나온대로 버젼이 맞춰지나보다.(잘 모름..)

그 다음 CUDA Toolkit을 설치한다. 
아래 사이트에서 본인의 운영체제에 맞게 선택해서 잘 설치하자. 
설치할때 Installer Type은 보통 runfile로 한다.(잘 모름..)

[https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive) 

이떄 조심해야하는게 설치를 하다보면 아래 그림에서 driver를 해제를 해주고 CUDA Tookit만 체크를 하는게 좋다. 연구실 선배말로는 driver를 여기서 체크하면 충돌이 일어난대요. 
![image](https://github.com/passion3659/passion3659.github.io/assets/89252263/4981ed5f-d637-45a8-9c5d-e0f8346fecbc)

자세하게는 다음 블로그를 참조해보자. [https://cow-kite24.tistory.com/331](https://cow-kite24.tistory.com/331) 

CUDA toolkit을 설치한 뒤에는 환경 변수를 추가하는 작업을 해야한다. 
이 과정은 구글에 CUDA Toolkit export라고 검색해서 하면 된다. 
sudo vi ~/.bashrc 이런 코드로 환경 열어서 export 어쩌고 코드 추가해주는것이다. 

그 뒤에 설치가 잘 됐고 잘 잡히는것을 보려면 유명한 다음 코드로 보면된다. 

```python
nvcc --version
```

아래 그림과 같이 나온다면 성공적으로 CUDA toolkit까지 설치가 된것이다!
![image](https://github.com/passion3659/passion3659.github.io/assets/89252263/544732d6-dc6c-4147-bd4a-0ce2e655bb8f)

# GPU ⇒ CUDA ⇒ PYTORCH 버젼 체크

보통 gpu를 사면 gpu 하드웨어의 버젼이 정해져있다. (RTX3090..이런느낌으로)

내가 가지고 있는 버젼이 RTX3090이다. 

그럼 [https://en.wikipedia.org/wiki/CUDA](https://en.wikipedia.org/wiki/CUDA) 를 봤을 때 

![image](https://github.com/daveredrum/ScanRefer/assets/89252263/4d7d5bb4-0731-4e3f-b9fa-8522ebc8f82c)

RTX3090은 은 8.6에 속하는 것을 볼 수 있다. [https://en.wikipedia.org/wiki/CUDA](https://en.wikipedia.org/wiki/CUDA) 에 아래 그림을 보면 쿠다 11.1부터 12.4까지는 호환이 가능하다고 나와있다. 그럼 난 CUDA 11.1부터 12.4까지 사용이 가능한 것이다.

![image](https://github.com/daveredrum/ScanRefer/assets/89252263/e10231c5-0d2a-4e2e-adc5-1471b14a7e6b)

내가 원하는 버젼을 [https://developer.nvidia.com/cuda-toolkit-archive](https://developer.nvidia.com/cuda-toolkit-archive) 에서 다운을 받고 그 다음 torch를 다운받아주는데 이때 버젼확인은 다음 사이트에서 하면 된다. <br>
[https://pytorch.org/get-started/previous-versions/](https://pytorch.org/get-started/previous-versions/)