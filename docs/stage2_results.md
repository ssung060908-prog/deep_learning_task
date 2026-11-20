# 2단계 — 전처리 비교 결과

실행일: 2026-11-17. 기본 6개 조합과 상위 2개 조합의 변수 곱·제곱 추가를 비교했습니다.
총 8개 설정 × 5-fold = 40회 CV 학습을 실제 제공 데이터에서 수행했습니다.

## 고정 조건

- 개발 6,316행, holdout 1,580행, 기존과 동일한 5-fold. 모델 seed=42, fold별 seed=42~46.
- 은닉층 [32], GELU, dropout 0.1, lr=0.0012204680039734717, weight_decay=6.030364379156746e-05, batch=256.
- class weighting 및 label smoothing 미사용. 최대 200 epochs, patience=25.
- 각 fold의 학습 부분으로만 전처리를 fit합니다. holdout AUROC는 계산하지 않았습니다.
- 변경한 변수: scaling, domain_features, 후속 비교에서 interactions만 변경했습니다.

## 결과

표의 표준편차는 5개 fold 점수의 표준편차입니다. 신뢰구간이나 seed 반복의 표준편차가 아닙니다.

| 순위 | 정규화 | 파생비율 | 곱·제곱 | 평균 CV AUROC | fold 표준편차 | 기준 대비 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | robust | 미사용 | 미사용 | 0.80535 | 0.01029 | +0.00000 |
| 2 | standard | 미사용 | 미사용 | 0.80430 | 0.00855 | -0.00104 |
| 3 | standard | 사용 | 미사용 | 0.80368 | 0.00834 | -0.00166 |
| 4 | robust | 사용 | 미사용 | 0.80356 | 0.01042 | -0.00179 |
| 5 | quantile | 미사용 | 미사용 | 0.80271 | 0.01038 | -0.00263 |
| 6 | quantile | 사용 | 미사용 | 0.80210 | 0.01024 | -0.00324 |
| 7 | standard | 미사용 | 사용 | 0.79139 | 0.00901 | -0.01396 |
| 8 | robust | 미사용 | 사용 | 0.79060 | 0.00906 | -0.01474 |

## 결정과 한계

3단계에 가져갈 전처리는 **RobustScaler + 파생비율 미사용 + 곱·제곱 미사용**으로 유지합니다.
최고 평균 CV AUROC는 0.80534535이고 기존 기준과 차이는 +0.00000000입니다.
전처리 개선을 확인한 것이 아니라 이번 후보 중 기준 설정보다 나은 평균 점수가 없었던 결과입니다.
StandardScaler + 파생비율 미사용은 차선 후보이며 변동성 확인은 5단계에서 진행합니다.
단일 초기화 seed와 현재 구조에서의 비교이므로 모든 모델에서 Robust가 우월하다고 결론 내리지 않습니다.
기본 6개 비교 후 상위 2개 조합에만 곱·제곱을 추가했고 두 경우 모두 평균 점수가 낮아 채택하지 않았습니다.
기존 기준의 각 fold 점수를 정확히 재현했고 데이터·분할·코드 해시와 기존 가중치 보존을 확인했습니다.

## 아웃풋과 다음 단계

- `configs/stage2_selected.json`: 3단계에서 고정할 전처리와 기준 모델 설정.
- `experiments/stage2/summary.json`, `summary.csv`: 8개 결과와 기준 대비 fold별 차이.
- `experiments/stage2/preprocessing/`, `interactions/`: 실행 조건·학습 곡선·원시 CV 결과.
- 새 제출용 가중치는 생성하지 않았습니다. 기존 `models/task1-v1/`은 그대로 유지합니다.
- 다음은 3단계: 선택한 전처리를 고정하고 [32], [64], [64,32] × ReLU/GELU 구조 비교입니다.
- 3단계 학습과 GitHub 업로드는 아직 하지 않았습니다.

## 재현 명령 (새 결과 폴더 필요)

```powershell
python experiment.py search --config configs/stage2_preprocessing.json --output runs/stage2_repeat --model-seed 42
python experiment.py search --config configs/stage2_interactions.json --output runs/stage2_interactions_repeat --model-seed 42
```

재현 실행은 필수가 아닙니다. 완료된 결과 파일을 그대로 사용하면 됩니다.
