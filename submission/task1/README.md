# Task 1 제출용 부분 묶음

이 폴더는 전체 팀 제출물이 아닙니다. task1.pt는 최종 전체 데이터 학습 모델입니다.

```powershell
python -m pip install -r requirements.txt
python code/predict.py --data test.csv --weights task1.pt --output predictions.csv
python code/task1.py --data 원본_heloc_release.csv --output retrained_task1.pt
```

학습 코드는 고정된 설정으로 전체 제공 데이터만 학습합니다. 후보 탐색을 다시 실행하지 않습니다.
원본 데이터는 배포하지 않습니다. 추론에는 23개 입력 열만 있으면 됩니다.
팀 전체 ZIP의 code/에 이 code/ 내용, 루트에 task1.pt와 병합된 requirements.txt를 넣으세요.
Task 2의 세 모델·학습 코드와 팀 정보가 들어간 최대 4쪽 report.pdf를 추가해야 합니다.
학습·검증 결과 및 보고서 원고는 저장소 docs/stage6_results.md에 있습니다.
