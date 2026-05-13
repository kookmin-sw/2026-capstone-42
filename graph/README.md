# graph

Citation graph 기반 후보 확장 및 feature engineering 모듈 (Module 3).

FAISS Retrieval 결과(`offline_output.json`)를 입력으로 받아 citation graph와 연결하고,
graph_score를 포함한 feature를 추출하여 `output_graph_stage3.json`을 생성합니다.

## 의존성 설치

    pip install -r requirements.txt

## 테스트 실행

    # graph/ 디렉터리에서 실행
    pytest tests/

## 평가 스크립트 실행

    # graph/ 디렉터리에서 실행
    python scripts/run_stage3_eval.py --smoke   # 앞 10개 query로 빠른 테스트
    python scripts/run_stage3_eval.py           # 전체 실행

## 필요한 데이터

`data/README.md` 참고
