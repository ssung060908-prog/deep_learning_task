# Task 1 v2 — 단일 HELOC DNN

Python 3.13에서 학습·검증했습니다. 원본 학습 데이터 없이 이 폴더만으로 예측할 수 있습니다.

```powershell
python -m pip install -r requirements.txt
python predict.py --data test.csv --weights task1.pt --output predictions.csv
```

입력: feature_names.json의 23개 이름을 가진 열. 열 순서는 바뀌어도 됩니다.
sample_id는 출력 식별자로만 사용하며 모델 입력이 아닙니다. target 열은 필요하지 않습니다.
출력: 입력 행 순서대로 sample_id, probability. probability는 P(is_at_risk=1)입니다.
Python 인터페이스: `from predict import predict_proba; probabilities = predict_proba(frame, "task1.pt")`.

모델: 전처리 후 116차원 → Linear(32) → GELU → Dropout(0.25) → Linear(1) → sigmoid.
가중치·전처리 통계·구조는 task1.pt 하나에 저장돼 있습니다. 앙상블이 아닙니다.
전체 7,896행, seed 42, 37 epochs로 학습했습니다.
3-seed 평균 CV AUROC: 0.80589589. 별도 개발 데이터 학습 모델의 holdout AUROC: 0.81421047.
이 점수는 전체 데이터 재학습 모델의 독립 평가 점수가 아닙니다. 비공개 Test 성능은 모릅니다.
holdout은 과거에도 관찰된 적이 있어 완전히 미관찰된 평가라고 설명하지 않습니다.
SHA256SUMS.json으로 파일 무결성을 확인할 수 있습니다.
