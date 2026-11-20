# 1단계 — 비교 조건 고정

2026-11-17. 과제 안내 PDF 8~19쪽과 HELOC 변수 설명서 전체를 다시 확인했습니다.
Task 1은 제공 데이터만 이용한 단일 완전연결 DNN, 출력 P(label=1), 평가 지표 AUROC입니다.
sample_id는 입력에서 제외하고 -7/-8/-9의 의미를 구분합니다.
앙상블과 다른 신경망을 이용한 전처리는 사용하지 않습니다.
교차검증·holdout·seed 분리는 과제의 지정 방식이 아니라 공정한 실험을 위한 설계입니다.

## 이번 구현

- 기존 `first_version.py`, `runs/task1/`, `models/task1-v1/`, 모델 ZIP은 변경하지 않습니다.
- 새로운 `experiment.py`가 비교 실험을 담당합니다.
- `prepare`는 기존 데이터 해시와 `split.csv`를 대조하고 동일한 분할과 행 순서를 고정합니다.
- 고정값: 개발 6,316행, holdout 1,580행, 5-fold, split_seed=42.
- 실제 분할 행 목록은 `runs/protocol_v1/protocol.json`에 저장하며 Git 추적 대상이 아닙니다.
- 원본 학습 결과의 파일 해시도 고정합니다. 원본 파일이나 데이터 순서·내용이 바뀌면 탐색을 거절합니다.
- 데이터 분할 seed는 고정 protocol에 들어 있습니다. 모델 초기화 seed는 `--model-seed`로 별도 지정합니다.
- `search`는 CV만 수행합니다. holdout 추론·AUROC 계산, 전체 데이터 재학습, 제출용 가중치 저장은 하지 않습니다.
- 설정 JSON은 단일 설정 객체 또는 여러 설정 객체의 목록을 지원합니다. 생략하면 고정된 기존 최선 설정을 사용합니다.
- 검증 비교에 필요한 설정, 각 fold 점수·학습 곡선, 소스 해시, 라이브러리 버전을 저장합니다.
- 비어 있지 않은 결과 폴더는 덮어쓰지 않습니다.

## 명령

과제 폴더에서 최초 1회 분할 고정(이미 이번 작업에서 실행):

```powershell
python experiment.py prepare
```

이후 비교 학습은 다음 방식으로 실행합니다. 아래 명령은 이번 작업에서 실행하지 않았습니다.

```powershell
python experiment.py search --output runs/baseline_cv_seed42 --model-seed 42
python experiment.py search --output runs/baseline_cv_seed123 --model-seed 123
```

두 명령은 같은 개발/holdout/fold 행을 사용하며 초기화 seed만 다릅니다.
fold별 학습 seed는 `model_seed + fold 번호(0부터)`입니다.
기본 최대 epoch=200, patience=25입니다. 최고 CV AUROC가 0.00001 초과 개선돼야 갱신됩니다.
별도 설정 파일 사용:

```powershell
python experiment.py search --config configs/baseline_v1.json --output runs/fixed_config_cv
```

현재 `selection.json`은 후보와 epoch를 고른 기록일 뿐 제출 모델이 아닙니다.
최종 평가·전체 재학습은 로드맵 6단계에서 설정을 확정한 뒤 진행합니다.
`first_version.py train`은 v1 당시의 탐색+holdout+재학습 전체 흐름이므로 후속 비교에는 사용하지 않습니다.

## 기존 결과 해석

기준 CV AUROC=0.8053453481, 당시 holdout AUROC=0.8139440453.
기존 holdout은 이미 관찰된 데이터이며, 다시 생성하거나 seed를 바꾼다고 관찰 이력이 사라지지 않습니다.
후속 설정 선택에 holdout 점수를 이용하지 않습니다.
최종 v1 가중치는 전체 제공 데이터로 학습됐으므로 CV 비교를 위해 이어 학습하거나 초기값으로 사용하지 않습니다.
각 CV 모델은 해당 fold 학습 데이터에서만 처음부터 학습합니다.

## 검증

```powershell
python -m unittest discover -s tests -v
```

테스트는 합성 데이터에서 짧은 CV 학습과 holdout 미사용, seed 분리, 분할/데이터 변조 거절,
원본 보호, 덮어쓰기 방지를 확인합니다. 합성 데이터는 실제 과제 학습에 섞지 않습니다.
