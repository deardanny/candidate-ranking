# 가상 사용자 3,000명 모의 A/B 테스트

**실제 사람·실제 클릭·LLM 응답은 각각 0개다.** 확률형 행동 시뮬레이터로 생성한 결과다.

기존 FairJob/EB-NeRD 모델을 재평가한 결과가 아니다. 가상 광고 환경에서 별도로 학습한 인기순과 문맥 LightGBM을 비교했다.

## 결과

| 실험 | A 선택률 | B 선택률 | B − A, %p | 조건부 95% 구간, %p |
|---|---:|---:|---:|---:|
| aa-check | 21.94% | 21.82% | -0.13 | [-1.31, +1.01] |
| ab-main | 21.94% | 30.96% | +9.02 | [+7.74, +10.28] |
| ab-null | 15.30% | 15.81% | +0.51 | [-0.48, +1.49] |

선택률은 방문당 모의 클릭 여부를 사용자별로 평균 낸 값이다. 각 사용자는 8회 방문한다. 구간은 사용자 단위 부트스트랩이며, 우리가 정의한 시뮬레이터 내부의 표본 변동만 나타낸다. 실제 사람에 대한 신뢰구간이 아니다.

- `aa-check`: 양쪽 모두 동일한 인기순 랭커. 같은 모델에서도 표본 차이는 발생할 수 있다.
- `ab-main`: 인기순과 문맥 LightGBM 비교. 사용자 취향·제약을 반영하는 반응 모델.
- `ab-null`: 같은 두 랭커를 비교하되 반응이 광고 정체성과 무관하도록 설정한 음성 대조군.

세 실험은 같은 3,000명을 재사용한다. 9,000명의 독립된 사용자가 아니다. 각 조건의 결과를 보고 시뮬레이터나 모델을 재조정하지 않는다.

## 연결한 실행 경로

`가상 사용자 → 실제 HTTP /rank → 사용자별 고정 배정 → 모델 추론 → SQLite 노출 스냅샷 → 별도 반응 생성기 → HTTP /events/feedback → 중복 제거·귀속 검증 → 사용자 단위 분석`

FastAPI, LightGBM, 기존 RequestSnapshot/ClosedFeedback 검증 계약을 사용했다. 서버가 클릭을 만들어 자기 성능을 평가하지 않도록 반응 생성은 클라이언트에서 실행한다. 요청·피드백은 실제 서버 시각으로 기록하고, 가상 방문 순서는 별도로 저장한다.

## 실행 무결성

- 전체 통과: **True**
- 사용자 배정: {'B': 1520, 'A': 1480}
- DB 기록: {'assignments': 9000, 'requests': 72000, 'feedback': 72000}
- 닫힌 피드백 검증: 72,000건
- 요청·피드백 재전송 중복 방지 확인: 90쌍
- HTTP/직접 추론 일치: 120건
- 클라이언트 실패: 0건
- 사용자 수 비율 검사 p: 0.4764
- HTTP 실험 소요 시간: 172.0초

서버와 생성기는 같은 컴퓨터에서 실행했다. 이 시간과 지연은 폐회로 모의 실행 결과이며, 운영 트래픽 처리 용량 측정이 아니다. 노출을 받지 못하거나 피드백이 빠진 방문은 무클릭으로 채우지 않는다. 피드백이 불완전하면 효과 계산을 보류한다.

## 해석 한계

- Preferences, examination, attrition and click probabilities are invented and not calibrated to people.
- Training and evaluation share the simulator, so model compatibility can produce an apparent lift by construction.
- Three experiments reuse the same 3000 personas; they are not 9000 independent users.
- Simulated cold ads are new IDs from the same invented feature distribution, not real first-exposure evidence.
- No bids, payment, conversion, advertiser revenue, real browser visibility or real human clicks are measured.
- Synthetic feedback never updates the existing FairJob/EB-NeRD production or research artifacts.

시뮬레이터를 바꿔도 효과가 그대로 유지되는지 보는 것이 중요하다. 클릭 수가 늘었다는 사실만으로 실제 광고 효과를 주장하지 않는다. 이 실험에서 확인한 결과물은 재현 가능한 실험 시스템과 설정한 환경에서의 정책 차이다.

설계 참고: [RecSim](https://arxiv.org/abs/1909.04847)은 사용자 상태와 반응 가정을 명시적으로 조절하는 추천 시뮬레이션을 다룬다. 이 구현은 RecSim 라이브러리를 사용하거나 인간 행동으로 보정한 모델이 아니다.

[정확한 집계·검증 결과](results.json) · [실행 및 API 안내](../../docs/SYNTHETIC_AB.md)
