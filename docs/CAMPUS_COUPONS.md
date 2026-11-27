# 캠퍼스 협찬 쿠폰: 클릭 목표와 이용 목표 비교

대학 앱의 광고 한 자리에 학교 주변 매장 쿠폰을 보여주는 가상 서비스다. 클릭을 늘리는 광고가 실제 이용에도 도움이 되는지 확인하려고 노출 이후의 쿠폰 사용을 별도 이벤트로 만들었다. 광고 24개와 사용자 3,000명은 전부 가상이며 LLM 호출이나 실제 매장 방문은 없다.

이전의 인기순 대 문맥 모델 실험과 달리, 이번에는 **같은 피처와 같은 훈련 노출에서 학습 목표만 바꾼 두 LightGBM**을 비교한다. A는 클릭, B는 노출 뒤 클릭하고 쿠폰까지 사용한 정답을 학습한다. B는 클릭 표본만 학습하는 조건부 CVR 모델이 아니며 ESMM을 구현한 것도 아니다.

```sh
python3 scripts/run.py -m fairjob_ranker.simulation.campus_run \
  --out runs/campus-coupons-v1 --report reports/campus-coupons-v1
```

기존 출력이 있으면 중단한다. 재실행 시 새로운 출력·보고서 경로를 사용한다. 별도 uvicorn 자식 프로세스가 loopback 포트를 열므로 제한된 실행 환경에서는 로컬 소켓 권한이 필요할 수 있다. 종료 시 소유한 서버만 정지한다.

기존 `Store`, 사용자 고정 배정, 요청 피처 스냅샷, 클릭 귀속 검증을 재사용한다. 추가한 `/events/coupon`은 클릭한 광고에만 쿠폰 사용을 귀속하고 재전송을 중복 제거한다. 피드백이 아예 없으면 무사용으로 집계하지 않는다. 24시간 관측 창은 가상 시간을 가속해 즉시 닫고, `virtual_redemption_delay_minutes`와 실제 서버 수신 시각을 분리한다. 실제 24시간 관측을 수행했다고 주장하지 않는다.

훈련 2,400명, 검증 600명, 평가 3,000명의 ID와 난수 계열을 분리한다. 같은 3,000명을 A/A·본 A/B·이동 제약 제거 조건에 재사용해 총 72,000회 방문을 처리한다. 그룹은 사용자에 고정되며, 각 방문의 후보 8개는 어느 정책에서도 같다. 모델과 행동 규칙은 평가 전에 고정한다.

주 지표는 전체 예정 방문당 쿠폰 사용률이며 사용자 단위 부트스트랩을 사용한다. 클릭당 쿠폰 사용률은 정책이 클릭할 사람 자체를 바꾸므로 보조 설명에만 사용한다. 거래 금액·광고 수익·실제 추가 방문은 측정하지 않는다.

```sh
python3 scripts/run.py -m fairjob_ranker.simulation.campus_analysis \
  --run runs/campus-coupons-v1 --report reports/campus-coupons-v1
python3 scripts/run.py -m pytest tests/test_campus_coupons.py -q
```

[광고 예시·실제 집계·한계](../reports/campus-coupons-v1/README.md) · [정확한 값과 검증 기록](../reports/campus-coupons-v1/results.json)
