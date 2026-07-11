# semiconductor-process-simulator

반도체 웨이퍼 공정 시뮬레이터와, 그 데이터를 학습해 EDS처럼 양품/불량을 스스로
판정하는 머신러닝 분류기. 시뮬레이터부터 퍼셉트론·MLP까지 전부 numpy로 밑바닥부터
구현했습니다.

**목적** — 실제 fab에서는 웨이퍼도 공정도 비싸서 라벨된 데이터(특히 희귀한 불량)를
모으기 어렵습니다. 이 프로젝트는 물리 시뮬레이터(디지털 트윈)로 데이터를 값싸게
대량 생성해 ML을 학습시키는 접근이 성립하는지를, 밑바닥 구현으로 끝까지
검증합니다 — 마지막 [sim2real 실험](#sim2real-가상-데이터는-진짜-fab에서-통하는가)에서
"합성 데이터 사전학습 + 소량 실데이터"가 실데이터만 쓰는 것보다 나은지 직접 측정합니다.

## 핵심 스토리

1. **가상 fab** — 실행할 때마다 공정 조건(Vth/Oxide/Leakage/CD/Temp의 중심·산포)이
   범위 안에서 자동 샘플링되고, 물리 상관관계(Oxide 두께·CD → Vth·Leakage,
   Temp → Leakage)가 반영된 웨이퍼 2만 장이 생성됩니다.
2. **현실적인 라벨** — 양불 판정(Result)은 **참값** 기준으로 내려지지만, 데이터에
   저장되는 것은 **측정값(참값 + 센서 노이즈)**입니다. 스펙 경계 근처에서는 측정만
   으로 양불을 확정할 수 없으므로, 어떤 모델도 100%에 도달할 수 없는 진짜 확률적
   분류 문제가 됩니다.
3. **단층 퍼셉트론의 한계** — 양품 영역은 스펙 박스(구간 AND)인데 퍼셉트론은 직선
   하나만 그을 수 있어 원리적으로 이 문제를 못 풉니다. 학습 중 오분류 수정 횟수가
   끝까지 줄지 않는 것(비수렴)으로 이를 실험적으로 확인했습니다.
4. **MLP로 돌파** — 은닉 뉴런들이 스펙 경계를 하나씩 학습하고 출력 뉴런이 이를
   조합해 박스를 표현합니다. 역전파까지 numpy로 직접 구현했습니다.

## 결과

평가는 **run 단위 분리**(학습 9 run / 테스트 2 run)로, 학습 때 본 적 없는 공정
조건에 대한 일반화 성능을 측정합니다. 데이터: 11 run × 20,000 = 220,000 웨이퍼.
특징은 5개(Vth, Oxide, Leakage, CD, Temp)이고, 양품 영역은 5차원 스펙 박스입니다.

| 지표 (불량 = positive) | 단층 퍼셉트론 (pocket) | MLP (은닉 16) |
|---|---|---|
| Test 정확도 | 79.86% | **83.82%** |
| Precision | 34.63% | 55.22% |
| Recall | **4.95%** | **85.44%** |
| F1 | 8.66% | **67.08%** |

("전부 양품" 베이스라인 정확도 80.70% — 불균형 데이터라 정확도만으로는 판단 불가)

퍼셉트론은 **불량의 95%를 놓치고**, 정확도(79.86%)마저 "전부 양품"(80.70%)보다
낮습니다. 특징이 5개로 늘어 스펙 박스가 5차원이 될수록, 직선 하나로 "가운데
구간만 양품"을 표현할 수 없다는 한계가 더 뚜렷해집니다. 아래 그림이 그 이유를
보여줍니다:

![decision boundary](docs/decision_boundary.png)

왼쪽(퍼셉트론)은 평면을 대각선으로 자를 뿐이라 오른쪽 위의 불량들을 전부 양품
판정하는 반면, 오른쪽(MLP)은 실제 스펙 박스(점선)에 가까운 닫힌 영역을 학습했습니다.

MLP는 P(양품)를 출력하므로 판정 컷오프를 조절해 운영점을 고를 수 있습니다.
불량 유출이 치명적인 fab이라면 컷오프를 올려 recall을 우선합니다:

| cutoff | precision(fail) | recall(fail) | F1 |
|---|---|---|---|
| 0.1 | 79.94% | 52.60% | 63.45% |
| 0.5 | 55.22% | 85.44% | 67.08% |
| 0.9 | 33.67% | **98.11%** | 50.13% |

## 전체 운영점 비교 (ROC · PR)

컷오프 몇 개가 아니라 **모든 임계값**에서의 판별력을 보려면 ROC·PR 곡선을 봅니다
(`python3 ml/visualize_metrics.py`). 퍼셉트론은 확률이 없으므로 경계까지의 부호
거리를 점수로 씁니다.

![model evaluation](docs/model_evaluation.png)

| 지표 | 단층 퍼셉트론 | MLP |
|---|---|---|
| ROC AUC | 0.386 | **0.925** |
| PR AP (불량) | 0.187 | **0.774** |

퍼셉트론의 ROC AUC가 **0.5(무작위)보다도 낮습니다**. 불량이 5차원 박스의 모든
면에 흩어져 있어, 하나의 선형 점수로는 불량의 순위조차 매길 수 없기 때문입니다
(한쪽 끝을 불량으로 몰면 반대쪽 끝의 불량을 놓칩니다). 오른쪽 아래 학습 곡선은
train 손실과 test 손실을 함께 그려, 과적합 여부를 눈으로 확인할 수 있게 합니다.

## 파생 특징: 표현력은 특징에서도 온다

스펙 한계는 비밀이 아니므로(fab은 자기 제품 요구사항을 압니다), 각 측정값이
**가장 가까운 스펙 경계에서 얼마나 떨어졌는지**(margin, 스펙 반폭으로 정규화)를
특징으로 직접 계산해 줄 수 있습니다 (`ml/features.py`, `--margins` 플래그).
여기에 margin들의 최솟값(MinMargin)까지 주면 "모든 margin ≥ 0 = 양품"이라는
박스 규칙이 **축 하나의 선형 컷**으로 펴집니다:

```bash
python3 ml/perceptron.py --margins
python3 ml/mlp.py --margins          # 저장된 모델에 플래그가 기록되어 judge가 자동 적용
```

| 모델 | 특징 | recall(fail) | F1 |
|---|---|---|---|
| 퍼셉트론 | 원본 5개 | 4.95% | 8.66% |
| 퍼셉트론 | + margin 6개 | **74.57%** | **69.73%** |
| MLP | 원본 5개 | 85.44% | 67.08% |
| MLP | + margin 6개 | 87.34% | 68.71% |

박스를 원리적으로 못 풀던 퍼셉트론이 MLP에 근접하게 살아납니다. 실제로 학습된
가중치를 보면 **MinMargin에 가장 큰 가중치(+0.149)** 를 스스로 부여합니다 —
규칙의 기하를 특징에 인코딩해 주면, 판단만 남기 때문입니다. 반면 MLP는 은닉층이
이미 박스를 표현할 수 있어 이득이 작습니다. 100%에 못 가는 건 여전합니다:
margin은 **측정값**으로 계산되지만 라벨은 **참값** 기준이라, 경계 근처의
모호함은 특징을 아무리 잘 만들어도 사라지지 않습니다.

## sim2real: 가상 데이터는 진짜 fab에서 통하는가

여기까지는 학습도 평가도 같은 시뮬레이터 안이었습니다 — 시뮬레이터가 곧 현실이니
잘 맞는 게 당연합니다. 진짜 질문은 **시뮬레이터의 물리가 현실과 조금 다를 때**도
가상 데이터가 쓸모 있는가입니다. 그래서 "현실 fab"을 하나 더 만들었습니다: 물리
결합 계수가 다르고(트윈은 이를 모름) 테스터 노이즈가 ~40% 더 큰 세계입니다
(`main.py --real` → `graph/real/run_XXX`, `simulation/config.py`의
`REAL_FAB_PHYSICS`/`REAL_FAB_NOISE`). 스펙 한계는 제품 요구사항이므로 동일합니다.

세 가지 전략을 현실 fab의 held-out run 2개(40,000장)로 채점합니다
(`python3 ml/sim2real.py`):

| 전략 | 실데이터 250장 | 1,000장 | 4,000장 | 16,000장 |
|---|---|---|---|---|
| scratch (실데이터만) | 57.44% | 69.88% | 76.15% | 77.45% |
| **fine-tune (합성 사전학습 + 실데이터)** | **78.61%** | **78.64%** | **79.37%** | **79.27%** |

zero-shot(합성만, 적응 없음) F1 77.13% · oracle(실데이터 20,000장 전부) 76.56%.

![sim2real](docs/sim2real.png)

- **실데이터 250장 + 합성 사전학습이, 실데이터 20,000장 scratch보다 낫습니다**
  (78.6% vs 76.6%) — 실데이터 요구량이 사실상 80분의 1로 줄어든 셈입니다.
- 이유는 양이 아니라 **다양성**입니다: 실데이터 pool은 공정 조건 1개(run 1개)에서
  나오지만, 트윈은 조건 12가지를 공짜로 휩쓸어 봤습니다. 실제 fab에서도 데이터
  수집의 병목은 장수보다 조건 커버리지입니다.
- fine-tune은 zero-shot보다 모든 구간에서 우위(+1.5~2.2%p) — 물리가 어긋난 만큼은
  소량의 실데이터 적응으로 메워집니다.

## 구조

```
semiconductor-process-simulator/
├── main.py                     # 시뮬레이션 엔트리포인트
├── simulation/                 # 가상 fab (데이터 생성)
│   ├── config.py               # 공정 스펙, 샘플링 범위, 측정 노이즈
│   ├── config_sampler.py       # run마다 공정 조건 자동 샘플링 (seed 기록)
│   ├── wafer_generate.py       # 웨이퍼 물리량 생성 + 측정 노이즈
│   ├── wafer_analysis.py       # 스펙 판정 → 수율
│   ├── defect_analysis.py      # 불량 원인별 집계
│   ├── visualization.py        # 분포/파레토 차트
│   ├── correlation_analysis.py # 상관 분석/산점도
│   ├── run_logger.py           # run_info.json + wafers.csv.gz 저장
│   └── run_manage.py           # run 폴더 관리 (run_001, run_002, ...)
├── ml/                         # 머신러닝 (numpy 밑바닥 구현)
│   ├── dataset.py              # run 취합 로더, run 단위 train/test 분리
│   ├── perceptron.py           # 단층 퍼셉트론 + pocket 알고리즘
│   ├── mlp.py                  # 다층 MLP + 역전파 + 가중 손실 + L2 + CLI
│   ├── features.py             # 파생 특징 (스펙 경계까지 margin, MinMargin)
│   ├── metrics.py              # 공용 지표 (혼동행렬/PRF1, ROC·PR 곡선)
│   ├── judge.py                # 저장된 모델로 새 run 판정 (EDS 단계)
│   ├── sim2real.py             # 합성 사전학습 vs 실데이터 예산 실험
│   ├── visualize_boundary.py   # 결정 경계 vs 실제 스펙 박스 시각화
│   └── visualize_metrics.py    # ROC·PR·학습곡선·혼동행렬 시각화
└── graph/                      # 실행 결과물 (gitignore)
    ├── run_XXX/                # run별 그래프, run_info.json, wafers.csv.gz
    └── real/run_XXX/           # "현실 fab" run (main.py --real)
```

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 실행

### 한 방에 (데이터 생성 → 학습 → 판정)

```bash
python3 run_all.py                 # 10 run 생성 → MLP 학습·저장 → 새 lot 판정
python3 run_all.py --runs 20       # 생성 run 수 조절
python3 run_all.py --cutoff 0.9    # recall 우선 운영점으로 판정
```

### 단계별로

```bash
# 1) 데이터 생성 - 돌릴 때마다 새로운 공정 조건으로 2만 장씩 축적
python3 main.py

# 2) 쌓인 데이터 현황
python3 ml/dataset.py

# 3) 학습/평가 (mlp.py는 학습된 모델을 graph/ml/mlp_model.npz로 저장)
python3 ml/perceptron.py
python3 ml/mlp.py

# 4) 학습된 모델로 새 웨이퍼 lot 판정 (EDS 단계)
python3 main.py                  # 새 run 생성 (예: run_011)
python3 ml/judge.py run_011      # 저장된 모델이 판정, 규칙 정답과 비교
python3 ml/judge.py run_011 0.9  # 컷오프를 올려 recall 우선 운영

# 5) 결정 경계 그림 생성 (graph/ml/decision_boundary.png)
python3 ml/visualize_boundary.py
```

## 비용 비대칭 반영 (가중 손실)

불량 유출(FN)이 양품 폐기(FP)보다 비싼 fab을 위해, 손실 함수에서 불량 샘플에
가중치를 줍니다 (`MLP(fail_weight=5)`). 학습 자체가 불량에 신중해져서, 같은 컷오프
0.5에서 불량 recall이 60.77% → 85.44%로 오릅니다 (precision을 내주는 의도된
트레이드오프). 추론 시점의 컷오프까지 합쳐 운영점을 정하는 손잡이가 둘이 됩니다.

## 다음 계획

- PyTorch 재구현과 numpy 구현 비교
- seed 다중 학습으로 지표 분산(신뢰구간) 리포트
