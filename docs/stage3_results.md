# 3단계 — DNN 구조 비교 결과

2026-11-19 완료. 전처리와 학습 설정을 고정하고 [32], [64], [64,32] × ReLU/GELU의 6개 구조를 비교했습니다.
신규 5개 × 5-fold = 25회 학습, 동일 조건의 기존 [32]+GELU 5-fold 결과는 재사용했습니다.
재사용 전에 protocol, 데이터·코드 해시, 라이브러리 버전, seed, epochs, patience, device, threads를 모두 대조했습니다.

## 고정 조건

- RobustScaler, 파생비율 및 곱·제곱 미사용. 전처리 후 입력 116개.
- 동일한 개발 6,316행, holdout 1,580행, 5-fold. 모델 seed=42, fold별 seed=42~46.
- Dropout 0.1, lr=0.0012204680039734717, weight_decay=6.030364379156746e-05, batch=256.
- LayerNorm, class weighting, label smoothing 미사용. 최대 200 epochs, patience=25.
- holdout 평가 및 전체 데이터 재학습 없음. 기존 제출용 모델 보존.

## 결과

| 순위 | 은닉층 | 활성화 | 파라미터 수 | 평균 CV AUROC | fold 표준편차 | 기존 대비 | 계산 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [64, 32] | gelu | 9,601 | 0.80553 | 0.01028 | +0.00019 | 신규 |
| 2 | [64] | gelu | 7,553 | 0.80550 | 0.01010 | +0.00016 | 신규 |
| 3 | [32] | gelu | 3,777 | 0.80535 | 0.01029 | +0.00000 | 재사용 |
| 4 | [64] | relu | 7,553 | 0.80472 | 0.01015 | -0.00063 | 신규 |
| 5 | [64, 32] | relu | 9,601 | 0.80470 | 0.01046 | -0.00065 | 신규 |
| 6 | [32] | relu | 3,777 | 0.80463 | 0.00968 | -0.00072 | 신규 |

표준편차는 5개 fold 사이의 변동이며 seed 반복의 표준편차나 신뢰구간이 아닙니다.
평균 최고는 [64,32]+GELU의 0.80553206입니다.
기존 [32]+GELU 대비 개선은 +0.00018671로 작습니다.

## 결정

주력 구조는 **[32]+GELU**를 유지하고 **[64]+GELU**를 4단계 보조 후보로 남깁니다.
로드맵의 “0.001 미만 차이는 우열을 단정하지 않고, 비슷하면 단순한 구조 유지” 원칙을 적용했습니다.
이는 통계적 유의성 검정이 아니며 두 모델이 동일하다고 증명한 것도 아닙니다.
[64,32]+GELU는 평균 최고 후보로 기록하되 더 큰 구조로 바꿀 만큼의 개선 근거가 아직 부족합니다.
현재 결론은 고정된 학습 설정과 단일 초기화 seed에서의 비교입니다. 후보별 최적 설정이 같다는 뜻은 아닙니다.
여러 seed에서의 안정성 확인은 5단계에 남아 있습니다.

## 아웃풋

- `configs/stage3_architectures.json`: 전체 6개 구조.
- `configs/stage3_architectures_new.json`: 신규로 실행한 5개 구조.
- `configs/stage3_selected.json`: 유지할 주력 설정.
- `configs/stage3_candidates_for_tuning.json`: 4단계에 가져갈 주력·보조 설정.
- `experiments/stage3/summary.json`, `summary.csv`: 순위·파라미터 수·fold별 차이.
- `experiments/stage3/new_runs/`, `reused_baseline.json`: 새 실험과 재사용 근거.

3단계는 완료했으며 4단계 학습은 실행하지 않았습니다. 새 제출용 가중치도 만들지 않았습니다.
이번 변경은 로컬에만 저장했고 GitHub 업로드는 하지 않았습니다.

재현이 필요하면 새로운 출력 폴더를 지정합니다. 이미 완료한 결과를 재사용해도 됩니다.

```powershell
python experiment.py search --config configs/stage3_architectures_new.json --output runs/stage3_repeat --model-seed 42
```
