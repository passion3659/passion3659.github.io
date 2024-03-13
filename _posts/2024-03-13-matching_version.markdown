---
layout: post
title:  "Matching (pytorch-lightning) to (pytorch)"
date:   2024-03-13 20:21:34 +0900
use_math: true
categories:
  - Programming
---
# 파이토치 라이트닝(pytorch-lightning)을 파이토치(pytorch) 버젼에 맞게 설치하기

연구를 하다가 어떤 코드를 돌려서 확인하고 싶어서 깃허브에 나온대로 install의 지시사항을 따라갔는데 pytorch-lightning을 설치하는 과정에서 너무 삽질을 많이해서 이번 글을 포스팅 한다. 

# 문제점

원래 설치했던 torch는 1.9.1로  다음 코드로 설치 했었다. 

```python
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

그 다음 requirements.txt를 아무 생각없이 pip install을 했는데 원래 설치했던 torch 버젼이 2.2로 업데이트가 되었다. 그리고 심지어 cuda이 1.2에 알맞는 torch로 업데이트가 된것….

논문에서 요구하는 버젼도 그렇지만 코드를 실행할 때  MinkowskiEngine이나 torch-geometric호환성을 생각했을 때 torch의 버젼이 절대 바뀌면 안되는 구조였다. (torch2.2 버젼으로 많이 돌려봄 ㅠ)

그래서 설치과정을 다시 봤는데 requirements.txt에 pytorch-lightning가 설치가 될 때 torch버젼이 강제로 올라가는 것을 볼 수 있었다. 

# trial & error

아마 코드를 깃허브에 올릴 당시에는 그냥 설치해도 되었을 것이다. 하지만 시간이 지나고 pytorch-lightning이 많이 업데이트가 되어서 이렇게 된거같다. 그래서 requirements에서 pytorch-lightning을 지우고 따로 설치를 해주기로 했다. 

일단 그냥 설치할때 ==1.6.0 이렇게 명시를 해서 pip install를 했는데 이상하게도 이렇게 설치할 때도  torch가 2.2버젼으로 강제로 버젼이 올라가버린다. 

그래서 pytorch-lightning의 홈페이지에 들어가서 예전 버젼의 설치 코드가 따로 있는지 확인했는데 이상하게도 그런게 하나도 없었다.(나는 못찾았다….)

pip 말고 conda로 설치하는 게 있길래 아래 코드로 install을 해봤다. 

```python
conda install lightning -c conda-forge
```

이 코드는 다행히 torch 버젼은 안 바뀌는데 torch에서 cuda를 인식을 못한다. torch.cuda.is_available을 했는데 false가 나온다.

다른 여러가지도 시도해봤는데 다 안됐다. pytorch-lightning이 설치될 때 종속되는 패키지들이 많이 있는데 그 버젼들이 다 안 맞았다….

# 해결

며칠을 헤매다가 연구실 선배(존경하는 갓지수 선배님)에게 실례를 무릅쓰고 물어봤다.<br>
원래 처음에 torch 설치했던 코드에 뒤에 pytorch-lightning를 붙혀서 설치를 했는데 이 코드가 신기하게도 pytorch-lightning이 앞의 코드를 보고 알아서 종속성있게 설치되고 필요한 패키지도 알맞게 설치를 해주었다. **처음부터 아래 코드로 torch랑 pytorch-lightning을 같이 설치해주면 된다.**<br>
(코드 뒷부분에 pytorch-lightning가 있다.)



```python
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 pytorch-lightning -f https://download.pytorch.org/whl/torch_stable.html
```