from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from Voc import Voc
from Prepare import batch2TrainData, trimRareWords
from Model import EncoderRNN, LuongAttnDecoderRNN, trainIters, GreedySearchDecoder, evaluateInput
from Const import PAD_token, EOS_token, SOS_token, device
from convokit import Corpus, download

# when reload
# import importlib
# importlib.reload()

os.environ['KMP_DUPLICATE_LIB_OK']='True'

corpus_name = "movie-corpus"
corpus = None
if corpus is None:
    corpus = Corpus(filename=download(corpus_name))

ids = corpus.get_conversation_ids()

conversation_list = []

for id in ids:
    conversation_list.append(corpus.get_conversation(id))

# print(conversation_list[0])
# print(type(conversation_list[0]))

conversation_pairs = []
voc = Voc('movie-corpus')
for i, element in enumerate(conversation_list):
    ut_ids = conversation_list[i].get_utterance_ids()
    rows = []
    if i < 1000: # 갯수를 제한 -------------------------------------------  !!!
        for id in ut_ids:
            utt = conversation_list[0].get_utterance(id)
            rows.append(utt.text)
            voc.addSentence(utt.text)
            # print(f"utt : {utt.text}" )
            # conv = conversation_list[0].get_utterance(id).get_conversation()
            # print( f"conversation : {conv}" )
        conversation_pairs.append(rows)

print(f"conversation_pairs : {len(conversation_pairs)}")


""" 빈도수가 적은 단어를 제거 """
MIN_COUNT = 3    # 제외할 단어의 기준이 되는 등장 횟수
pairs = trimRareWords(voc, conversation_pairs, MIN_COUNT)


""" 모델 수행하기 """

# 모델을 설정합니다
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# 불러올 checkpoint를 설정합니다. 처음부터 시작할 때는 None으로 둡니다.
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# loadFilename이 제공되는 경우에는 모델을 불러옵니다
checkpoint = None
if loadFilename:
    # 모델을 학습할 때와 같은 기기에서 불러오는 경우
    checkpoint = torch.load(loadFilename)
    # GPU에서 학습한 모델을 CPU로 불러오는 경우
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# 단어 임베딩을 초기화합니다
embedding = nn.Embedding(voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# 인코더 및 디코더 모델을 초기화합니다
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# 적절한 디바이스를 사용합니다
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')




""" 학습 수행하기 """

# 학습 및 최적화 설정
clip = 50.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 4000
print_every = 1
save_every = 500

save_dir = './'

# Dropout 레이어를 학습 모드로 둡니다
encoder.train()
decoder.train()

# Optimizer를 초기화합니다
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)

# if loadFilename:
#     encoder_optimizer.load_state_dict(encoder_optimizer_sd)
#     decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# cuda가 있다면 cuda를 설정합니다
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# 학습 단계를 수행합니다
print("Starting Training!")
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, hidden_size, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename, checkpoint)



""" 평가 수행하기 """

# Dropout 레이어를 평가 모드로 설정합니다
encoder.eval()
decoder.eval()

# 탐색 모듈을 초기화합니다
searcher = GreedySearchDecoder(encoder, decoder)

# 채팅을 시작합니다 (다음 줄의 주석을 제거하면 시작해볼 수 있습니다)
evaluateInput(encoder, decoder, searcher, voc)