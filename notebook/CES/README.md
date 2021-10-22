# Timeseries Anomaly Detection
Timeseries Data Anomaly Detection

# Dacon
**HAI2021**

https://dacon.io/competitions/official/235757/leaderboard

# Dataset

**1. 학습 데이터셋 (6개)**

파일명: 'train1.csv', 'train2.csv', 'train3.csv', 'train4.csv', 'train5.csv', 'train6.csv'

설명: 정상적인 운영 상황에서 수집된 데이터(각 파일별로 시간 연속성을 가짐)

- Column1 ('timestamp'): 관측 시각

- Column2, 3, …, 80 ('C01', 'C02', …, 'C86'): 상태 관측 데이터


**2. 검증 데이터셋 (1개)**

파일명: 'validation.csv'

설명 : 5가지 공격 상황에서 수집된 데이터(시간 연속성을 가짐)

- Column1 ('timestamp'): 관측 시각
- Column2, 3, …, 80 ('C01', 'C02', …, 'C86'): 상태 관측 데이
- Column88: 공격 라벨 (정상:'0', 공격:'1')


**3. 테스트 데이터셋 (3개)**

파일명: 'test1.csv', 'test2.csv', 'test3.csv'

- Column1 ('timestamp'): 관측 시각
- Column2, 3, …, 80 ('C01', 'C02', …, 'C86'): 상태 관측 데이터


**4. sample_submission.csv(제출양식)**

Column1 ('timestamp'): 관측 시각
Column2 ('attack'): 공격 예측값(정상:'0', 공격:'1')


**5. eTaPR-21.8.2-py3-none-any.whl:** 

평가산식 도구
