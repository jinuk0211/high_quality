
llm 만들기 가이드 part 4.2 - 데이터 수집
map neo - 2024년 6월 3일
성능 - (llama 3 8b와 mistral 7b v0.2 사이 어딘가)
오픈소스 (weight만 x)
arxiv.org/pdf/2…

https://www.threads.net/@rien_n_est/post/C6ET8KAScM-

1. 필터링
=============================
bad_url_words.py - pornhub, masturbation,レイプ, 兽交, bdsm, 女の子, wank 등의 성적의미가 될수 있는 단어가 들어간 url 분리

filter.py
데이터셋, 파일, 줄단위, url
문장 수, 단어 수, 평균 단어 길이, 해시태그 관련 룰 설정
(e.g
rule 1: Number of sentences in a document should be > 1
rule 10: The fraction of punctuations in words must be > 0
)
fasttext로 점수

filter_for_redpajama.py
Spark를 사용하여 JSON 라인 형식의 문서 데이터와 관련 품질 신호(quality signals) 데이터를 읽고,
두 데이터를 join 필터링해 결과를 JSON 형식으로 저장

util.py, util_with_try.py
데이터별로 규칙 설정
e.g ) refinedweb, gopher, c4, rp-v1 등 

2.deduplication
==================================
minhashLsh.py
jaccard 유사성을 측정
Jaccard(A, B) = (A ∩ B)/(A ∪ B)
MinHash는 각 집합의 요소를 더 큰 숫자 도메인으로 매핑하는 여러 개의 서로 다른 해시 함수를 사용하는 것을 포함합니다.
각 집합에 대해 이 여러 해시 함수를 모든 요소에 적용하고, 각 해시 함수가 생성하는 가장 작은 해시 값을 해당 집합의 최소 해시 값으로 선택합니다. 
따라서 각 집합은 이러한 최소 해시 값의 벡터로 나타낼 수 있으며, 이는 집합의 MinHash signature 형성
텍스트 데이터의 경우, n-gram 접근법을 사용하여 집합을 구성
텍스트의 signature을 얻은 후, Locality-Sensitive Hashing (LSH) 를 사용하여 Jaccard 유사도가 특정 임계값을 초과하는 후보 집합 쌍을 빠르게 식별
이는 유사한 항목에 대한 검색 프로세스를 가속화
구체적인 접근법은 서명을 여러 band로 나누고, 각 band에 여러 해시 값을 포함시킵니다.
그런 다음 또 다른 해시 함수를 사용하여 각 band를 해시 버킷에 매핑
동일한 band 해시를 가진 모든 집합은 동일한 해시 버킷에 매핑
동일한 해시 버킷에 있는 모든 집합 쌍은 유사 후보 쌍으로 간주
여기에서는 signature을 구성하기 위해 128개의 고유 해시 함수를 사용하고, 이를 9개의 band로 나누며 각 band는 13개의 해시 값을 포함
Jaccard 임계값은 0.8로 설정
유사한 쌍을 식별한 후, 연결된 구성 요소(connected components)를 구성 
연결된 구성 요소 내에서 하나의 텍스트를 유지하고 다른 텍스트는 제거 
방대한 양의 데이터를 효율적으로 처리하기 위해 map-reduce 기반의 분산 사용

exact_substr_dedup.py
언어의 다양성을 고려할 때, 반복되는 텍스트의 길이가 길 경우, 이들은 서로 파생되었거나 동일한 출처에서 나온 것일 가능성이 매우 높음
따라서 두 텍스트 ti와 tj가 충분히 긴 부분 문자열을 공유할 때, 즉 t_a..a+k_i = t_b..b+k_j인 경우, 그 중 하나를 완전히 제거
길이 임계값은 arXiv:2107.06499, 2021의 정해진 값 사용, k=50으로 선택

분산 처리임으로 , 단일 노드의 메모리는 모든 데이터를 저장하기에 충분하지 않다 
따라서 위의 논문을 완전히 구현하지 않음. 대신, 우리는 각 텍스트를 1step에 50자 길이의 슬라이딩 윈도우로 분할 

그런 다음 각 윈도우에 대해 해당 문서 ID와 위치와 함께 SHA256 해시 값을 계산
이후 동일한 해시 값을 가진 window에 대해 첫 번째 window 제외하고 중복으로 표시 
마지막으로 텍스트 ID와 위치를 사용하여 원래 문자열을 복원하고, 중복 표시를 기준으로 삭제할지 여부를 결정

paragraph.py
문단 중복 제거

문단 중복 제거는 텍스트 내의 모든 중복 문단을 제거하는 UTF-8 줄바꿈 문자 "\n"로 구분
웹사이트 탐색 헤더, 광고 및 유사한 요소를 제거하는 데 효과적
텍스트의 일부를 삭제해서 콘텐츠 분석에 일부 방해가 발생할 수 있음
구현 방법은 먼저 텍스트를 UTF-8 줄바꿈 문자 "\n"를 사용하여 여러 문단으로 분할하고, 각 문단에 해당 문서 ID와 텍스트 내 위치(offset)를 태그로 추가
그런 다음 각 문단을 SHA256을 사용하여 해시. 이후 이 해시 값들로 중복 제거, 중복 제거가 완료되면 문서 ID와 offset를 기준으로 중복 제거된 텍스트를 복원

마크다운 포맷으로 전환
=====================================

![image](https://github.com/jinuk0211/ai_paper_review/assets/150532431/fd0f89fa-f81d-4832-8904-f7264470dae9)

pdf 같은 문서에서 텍스트를 뽑아내는 것은 힘들지만 좋은 정보들을 갖고있다

이슈) 다양한 레이아웃 
text, titles, captions, images, tables, and formulas와 같은 레이아웃 요소를 식별하고 관계를 분석해야한다.

PP-StructureV2, Marker4, Vary , Nougat 같은 오픈소스 모델등이 잇지만 별로 뛰어나지 않다

marker 여러 언어 제공하고, PP-StructureV2를 사용하여 효율적인 layout parsing
그림 4에서 설명된 것처럼, 여기서사용한 문서 변환 프레임워크는 Layout Detection, Element Recognition, Ordering, and Post Process 과정

layout detection 단계
--------------
FGD 알고리즘과 StructureV2를 사용해 header,footer,formulas, text 등으로 레이아웃 식별

element recognization 단계
-----------

요소별로 사용한 OCR 모델
formula
Pix2Text를 이용해 훈련한 TrOCR model 가 Latex-OCR ,Taxify의 formula 인식에 우수한 성능을 보임

text
PP-OCRv4

table
SLANet

headers, footers, 페이지 수는 버려짐
post process, 재구성 단계로 넘어가지지 않음
classifier

ordering 단계
-------------
블럭사이의 관계 또한 중요
다단, 페이지에 걸쳐있는 레이아웃 등을 탐지해내기 위해 (multi-column, cross page)
layoutlm v3 사용

postprocess 단계
--------------
줄바뀔때 하이픈 (e.g net-working)
OCR 텍스트 추출에서 한 문장이 여러 조각으로 쪼개졌을 때 이를 완전한 문장으로
마크다운으로 OCR된 수학공식에는 빠진요소, 잘못 추출된 기호
이를 위해  Qwen 7B모델을 파인 튜닝해서 (xi, yi)데이터에 대해서 지도학습 
x는 잘못된 수학공식, y는 올바르게 된 수학공식

inference 최적화로는 vLLM 사용(pagedattention, 양자화)
이 단계 프롬프트 예시 

![image](https://github.com/jinuk0211/ai_paper_review/assets/150532431/20970270-38ff-4d15-8107-7e9f503972b3)

위의 과정을 통해 OCR된 텍스트의 질을 크게 향상시킬수 있었음 또한 FastDeploy6를 사용해 효율 최대화

classifier
config.yaml
text_cls.py
utils.py

detector
config.yaml
text_detect.py
utils.py

latex
base_model.py
비젼트랜스포머(vit.py, hybrid.py, It_transformer.py)
latex_rec.py
setup.py

postprocessor
gg

recognizer
config.yaml
text_recognize.py
utils.py

structure
config.yaml
layout 요소들 txt파일
structure_doc.py
utils.py

table
gg

tool
gg

고품질 데이터
================

construct_fasttext_train_data.py

consturct_dataset.py

filter_data_final.py

quantize_fasttext_model.py

train_fasttext.py

utils.py
