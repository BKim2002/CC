import json
from pathlib import Path

json_path = Path(__file__).parent/"data"/"competencies.json"

# JSON 파일을 Python 딕셔너리로 변환
# registry가 JSON 정보를 담는 Python 딕셔너리
with json_path.open(encoding="utf-8") as file:
    registry = json.load(file)

# registry의 "items" key가 역량(dict)들의 list를 value로 가지고 있음
# 역검 종합점수와 역검 필기의 역량 총계 53개(상위 + 중위 + 하위 + 최하위) 및 영상면접 총계 9개(평가요인 + 세부항목) 전부 registry의 items 리스트에 각각의 dict로 저장되어 있음
# items로 꺼내오기
items = registry["items"]

# 검색용 딕셔너리 생성; 각 딕셔너리에 이름표 달아주기
# str("역량 이름") : {dict}(역량 정보)
lookup = {}     # 신역검 62개 역량명 + 3개 구역검 역량명 = 총합 65개
for item in items:
    # 역량 이름 = 역량 정보 매칭
    lookup[item["name"]] = item

    # 영상면접 구역검 세부항목 역량 명칭도 매칭; lookup["안정감"] == lookup["면접태도_안정성"]
    for alias in item["aliases"]:
        lookup[alias] = item