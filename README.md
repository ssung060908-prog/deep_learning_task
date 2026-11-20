## Task 1 v2 다운로드

- [고정 v2 모델 ZIP](https://github.com/ssung060908-prog/deep_learning_task/raw/refs/tags/task1-v2/models/task1-v2.zip): 모델만 받아 추론.
- [고정 v2 제출 부분 ZIP](https://github.com/ssung060908-prog/deep_learning_task/raw/refs/tags/task1-v2/submission/task1.zip): Task 1 학습 코드·가중치.
- [추가 개선 검토](docs/post_v2_improvement_plan.md): 새 학습 없이 조사한 후속 후보와 비교 예산.

v1과 v2를 별도로 보존합니다. ZIP에 원본 데이터는 포함하지 않습니다.
아래 단계별 '업로드 전' 문구는 당시 상태의 기록입니다.

## 최신 상태 — 2026-11-19, 6단계 완료

Task 1 최종 모델: `models/task1-v2/task1.pt` (단독 사용 ZIP: `models/task1-v2.zip`).
제출용 코드·가중치: `submission/task1.zip` (Task 1 부분만).
3-seed CV AUROC 0.80589589; 별도 개발 모델 holdout AUROC 0.81421047.
전체 7,896행, seed 42, 37 epochs. 최종 모델 재로딩·독립 추론·제출 학습 코드 재현 검증 완료.
상세 기록과 보고서 원고: `docs/stage6_results.md`. 이전 단계 기록은 아래에 보존합니다.
GitHub 업로드 및 전체 과제 제출은 아직 하지 않았습니다.

# Task 1 실행

## 최신 진행 — 5단계 완료 (2026-11-19)

3개 초기화 seed 비교 완료. 평균 최고는 후보 A, 작은 점수 차이에 대한 사전 단순성 기준에 따라 후보 B를 선택했습니다. 최종 계획은 seed 42, 37 epochs로 고정했습니다. [상세 결과](docs/stage5_results.md). 다음은 6단계이며 holdout 평가·전체 재학습은 아직 하지 않았습니다. 아래는 이전 단계 기록입니다.

## 최신 진행 — 4단계 완료 (2026-11-19)

12개 설정 비교 완료. 최고 CV AUROC 0.80600, v1 대비 +0.00066. 상위 2개를 5단계 안정성 확인 후보로 저장했습니다. [상세 결과](docs/stage4_results.md). holdout과 기존 가중치는 보존했고 5단계는 아직 실행하지 않았습니다. 아래는 이전 단계의 기록입니다.

## 최신 결과 — 3단계 완료 (2026-11-19)

구조 6개 비교를 완료했습니다(신규 25회 CV 학습, 동일 조건 5회 재사용). 평균 최고는 [64,32]+GELU 0.80553이지만 기존 대비 차이 0.00019가 작아 주력 [32]+GELU를 유지합니다. [64]+GELU는 다음 미세 조정의 보조 후보입니다. [상세 결과](docs/stage3_results.md). 다음은 4단계이며 아직 실행하지 않았습니다.

## 최신 결과 — 2단계 완료

전처리 8개 설정 × 5-fold 비교 결과, 기존 RobustScaler·파생변수 미사용을 유지합니다. CV AUROC 0.80535. [상세 결과](docs/stage2_results.md), [다음 단계용 설정](configs/stage2_selected.json). 새 제출 모델은 만들지 않았으며 다음은 3단계 구조 비교입니다.

## 후속 실험 (1단계 완료)

2026-11-17: 기존 모델과 결과를 보존하고, 같은 데이터 분할에서 비교하는 `experiment.py`를 추가했습니다.
실제 개발 6,316행/holdout 1,580행/5-fold 분할은 `runs/protocol_v1/protocol.json`에 고정돼 있습니다.
분할 고정은 완료했고 추가 실제 데이터 학습은 하지 않았습니다.

다음 비교를 실행할 때:

```powershell
python experiment.py search --output runs/baseline_cv_seed42 --model-seed 42
```

이 명령은 기존 최선 설정으로 CV만 계산합니다. holdout 평가, 전체 데이터 재학습,
제출용 가중치 생성은 하지 않습니다. 다른 초기화 seed를 써도 데이터 분할은 동일합니다.
설정 JSON 지정은 `--config configs/baseline_v1.json`으로 가능합니다.
같은 결과 폴더에는 다시 쓰지 않습니다. [1단계 상세 사용법](docs/stage1_protocol.md)을 참고하세요.

아래 `first_version.py train`은 최초 v1 학습 방식의 기록입니다. 후속 비교에는 위 명령을 사용하세요.

## 최초 v1 실행 방식 (기록)

이 폴더에 전용 Python 환경 `.venv`, 필요한 라이브러리와 실제 데이터가 준비되어 있습니다.
VS Code 실행 버튼을 누르거나 인수 없이 실행하면 사용법이 표시됩니다.
다른 Python으로 `first_version.py`를 실행해도 이 폴더의 `.venv`로 자동 전환합니다.

현재 폴더의 PowerShell에서 본 학습:

```powershell
python first_version.py train
```

기본은 24개 설정 × 5-fold, 최대 200 epoch입니다. 결과는 `runs/task1`에 저장됩니다.
다시 실행할 때는 `--output runs/second`처럼 새로운 결과 폴더를 지정하세요.

빠른 동작 확인:

```powershell
python first_version.py train --output runs/quick --trials 2 --folds 2 --epochs 5 --patience 2
```

데이터 경로를 생략하면 이 폴더의 `data/heloc_release.csv`를 사용합니다.
모델은 단일 DNN이며 전처리, 파생비율, 하이퍼파라미터를 CV AUROC로 비교합니다.
holdout은 설정 선택 이후 한 번 평가하고, 최종 모델은 전체 데이터로 재학습합니다.
제출용 모델과 전처리가 함께 `task1.pt`에 저장됩니다.

예측:

```powershell
python first_version.py predict --data data/test.csv --weights runs/task1/task1.pt --output predictions.csv
```

환경 복원: Python 환경을 다시 만든 뒤 `python -m pip install -r requirements.txt`를 실행하세요.
원래 코드는 `first_version.py.before-env-fix.bak`에 보존했습니다.


## GitHub에서 모델만 받기

학습이 끝난 v1 모델은 `models/task1-v1/`에 고정 보관합니다. 재학습 없이 사용하려면
[모델 전용 ZIP 다운로드](https://github.com/ssung060908-prog/deep_learning_task/raw/refs/heads/main/models/task1-v1.zip)를 받아 압축을 풀고 안의 README를 따르세요.

- `task1.pt`: 가중치 + 전처리 + 모델 구조
- `predict.py`: 독립 실행 가능한 예측 코드
- `requirements.txt`: 라이브러리 버전
- `feature_names.json`: 필요한 입력 열
- `metrics.json`: 학습·평가 요약
- `SHA256SUMS.json`: 파일 해시

최초 고정 버전의 Git 태그는 `task1-v1`입니다. 이후 main이 바뀌어도
[고정 버전 모델 ZIP](https://github.com/ssung060908-prog/deep_learning_task/raw/refs/tags/task1-v1/models/task1-v1.zip)을 사용할 수 있습니다.

현재 모델: 은닉층 [32], GELU, Robust scaling. CV AUROC 0.80535, holdout AUROC 0.81394.
가중치는 전체 데이터 40 epochs 재학습 결과이며 holdout 점수는 개발 데이터만으로 학습한 평가 모델의 결과입니다.
private Test 점수는 아직 모릅니다.

`experiments/task1-v1/`에는 실험 요약, `docs/model_design_roadmap.md`에는 후속 설계 계획이 있습니다.
원본 과제 데이터는 공유하지 않습니다. 새 컴퓨터에서 재학습하려면 허가받은 데이터를 별도로
`data/heloc_release.csv`에 준비하고 필요한 라이브러리를 설치해야 합니다.
이 저장소는 Task 1 결과이며 Task 2와 보고서를 포함한 전체 과제 제출물은 아닙니다.
