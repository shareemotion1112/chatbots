# 기본 단어 토큰 값
PAD_token = 0  # 짧은 문장을 채울(패딩, PADding) 때 사용할 제로 토큰
SOS_token = 1  # 문장의 시작(SOS, Start Of Sentence)을 나타내는 토큰
EOS_token = 2  # 문장의 끝(EOS, End Of Sentence)을 나태는 토큰
MAX_LENGTH = 10  # 고려할 문장의 최대 길이



import torch
import platform

def get_device():
    device = None
    myOs = platform.platform()
    if myOs.split('-')[2] == 'arm64':
        # device = 'mps'
        device = 'cpu'
    else:
        USE_CUDA = torch.cuda.is_available()
        device = torch.device("cuda" if USE_CUDA else "cpu")
    return device

device = get_device()
print(f"device : {device}")