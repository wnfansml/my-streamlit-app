import json
import re
import urllib.parse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from openai import OpenAI


# ---------------------------
# Config / UI
# ---------------------------
st.set_page_config(page_title="🎁 GiftMatch", page_icon="🎁", layout="wide")

st.title("🎁 GiftMatch — 상황/관계 기반 선물 추천 (실제 상품 링크 연결)")
st.caption("관계·친밀도·예산·상황을 입력하면 추천 3개를 순위/점수 근거로 제시하고, 네이버 쇼핑 상품 링크로 바로 연결합니다.")

with st.sidebar:
    st.header("🔑 API 키 설정 (배포용)")
    openai_key = st.text_input("OpenAI API Key", type="password", key="openai_key")
    naver_client_id = st.text_input("Naver Client ID", type="password", key="naver_id")
    naver_client_secret = st.text_input("Naver Client Secret", type="password", key="naver_secret")


    st.divider()
    st.markdown("**✅ 네이버 쇼핑 검색 API**는 요청 헤더에 Client ID/Secret을 넣어 호출합니다. :contentReference[oaicite:1]{index=1}")
    st.markdown("OpenAI 키와 네이버 키를 모두 넣어야 추천+상품 연결이 동작합니다.")

# Session state init
if "wishlist" not in st.session_state:
    st.session_state["wishlist"] = []  # list of rec dict
if "feedback" not in st.session_state:
    st.session_state["feedback"] = []  # list of feedback records
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None


# ---------------------------
# Helpers
# ---------------------------
@dataclass
class RecItem:
    rank: int
    title: str
    score: int
    rationale: str
    price_hint: str
    search_query: str
    alt_ideas: List[str]


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<.*?>", "", text)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default


def make_naver_shop_search_url(query: str) -> str:
    # Naver shopping web search URL (for fallback)
    q = urllib.parse.quote(query)
    return f"https://search.shopping.naver.com/search/all?query={q}"


def make_naver_map_search_url(query: str) -> str:
    # Nearby store search fallback (Naver Map)
    q = urllib.parse.quote(query)
    return f"https://map.naver.com/p/search/{q}"


def within_budget(price: int, budget: int, tolerance: float = 0.25) -> bool:
    """Allow +/- tolerance range around budget for 'reasonable' results."""
    if budget <= 0 or price <= 0:
        return True
    low = int(budget * (1 - tolerance))
    high = int(budget * (1 + tolerance))
    return low <= price <= high


# ---------------------------
# Naver Shopping Search API
# ---------------------------
@st.cache_data(ttl=60 * 30, show_spinner=False)
def naver_shop_search(
    query: str,
    client_id: str,
    client_secret: str,
    display: int = 20,
    start: int = 1,
    sort: str = "sim",   # sim/date/asc/dsc :contentReference[oaicite:2]{index=2}
    exclude_used: bool = True,
) -> Dict[str, Any]:
    """
    Calls Naver Shopping Search API (JSON).
    Official endpoint: https://openapi.naver.com/v1/search/shop.json :contentReference[oaicite:3]{index=3}
    """
    url = "https://openapi.naver.com/v1/search/shop.json"
    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }
    params = {
        "query": query,
        "display": display,
        "start": start,
        "sort": sort,
    }
    # exclude option exists in doc; exclude=used:cbshop 형태 :contentReference[oaicite:4]{index=4}
    if exclude_used:
        params["exclude"] = "used:cbshop:rental"

    resp = requests.get(url, headers=headers, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def pick_best_items(api_json: Dict[str, Any], budget: int, top_k: int = 3) -> List[Dict[str, Any]]:
    items = api_json.get("items", []) or []
    cleaned: List[Dict[str, Any]] = []
    for it in items:
        title = _strip_html(it.get("title", ""))
        link = it.get("link", "")
        image = it.get("image", "")
        mall = it.get("mallName", "")
        lprice = _safe_int(it.get("lprice", 0), 0)
        cat1 = it.get("category1", "")
        cat2 = it.get("category2", "")
        cat3 = it.get("category3", "")

        # Budget filtering (client-side, since API spec doesn't provide min/max price params) :contentReference[oaicite:5]{index=5}
        if budget > 0 and lprice > 0 and not within_budget(lprice, budget, tolerance=0.35):
            continue

        cleaned.append(
            {
                "title": title,
                "link": link,
                "image": image,
                "mallName": mall,
                "lprice": lprice,
                "category": " > ".join([c for c in [cat1, cat2, cat3] if c]),
            }
        )

    # Prefer cheaper if many; still keep order reasonable
    cleaned.sort(key=lambda x: (x["lprice"] if x["lprice"] > 0 else 10**12))
    return cleaned[:top_k]


# ---------------------------
# OpenAI Recommendation
# ---------------------------
def build_prompt(payload: Dict[str, Any]) -> Tuple[str, str]:
    system = (
        "너는 20대 라이프스타일을 이해하는 '선물 추천 전문가'다.\n"
        "사용자 입력(관계/친밀도/예산/상황/취향/전달방식/키워드)을 바탕으로 선물 3가지를 '추천 순위(1~3위)'로 제안하라.\n"
        "각 추천은 점수(0~100), 점수 근거(1문장), 예상 가격대(문자열), 네이버 쇼핑 검색에 사용할 검색 쿼리(짧고 구체적), 대체 아이디어 2~3개를 포함하라.\n"
        "출력은 반드시 JSON만. 다른 텍스트 금지."
    )

    user = (
        "다음 입력을 바탕으로 JSON 형식으로만 추천해줘.\n"
        f"입력: {json.dumps(payload, ensure_ascii=False)}\n\n"
        "출력 JSON 스키마(반드시 준수):\n"
        "{\n"
        '  "recommendations": [\n'
        "    {\n"
        '      "rank": 1,\n'
        '      "title": "선물명",\n'
        '      "score": 0,\n'
        '      "rationale": "점수 근거 한 문장",\n'
        '      "price_hint": "예: 2~3만원",\n'
        '      "search_query": "네이버 쇼핑 검색어(구체적)",\n'
        '      "alt_ideas": ["대체안1","대체안2","대체안3"]\n'
        "    }\n"
        "  ]\n"
        "}\n"
    )
    return system, user


def parse_recs(raw: str) -> List[RecItem]:
    # Extract JSON robustly
    raw = raw.strip()
    # Try direct JSON
    try:
        obj = json.loads(raw)
    except Exception:
        # Fallback: find first {...} block
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            raise ValueError("모델 응답에서 JSON을 찾지 못했습니다.")
        obj = json.loads(m.group(0))

    recs = obj.get("recommendations", [])
    if not isinstance(recs, list) or len(recs) == 0:
        raise ValueError("recommendations 배열이 비어있습니다.")

    parsed: List[RecItem] = []
    for r in recs:
        parsed.append(
            RecItem(
                rank=_safe_int(r.get("rank", 0), 0),
                title=str(r.get("title", "")).strip(),
                score=_safe_int(r.get("score", 0), 0),
                rationale=str(r.get("rationale", "")).strip(),
                price_hint=str(r.get("price_hint", "")).strip(),
                search_query=str(r.get("search_query", "")).strip(),
                alt_ideas=list(r.get("alt_ideas", []) or []),
            )
        )
    # Sort by rank then score
    parsed.sort(key=lambda x: (x.rank if x.rank > 0 else 999, -x.score))
    return parsed[:3]


def get_recommendations(openai_api_key: str, payload: Dict[str, Any]) -> List[RecItem]:
    client = OpenAI(api_key=openai_api_key)
    system, user = build_prompt(payload)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
    )
    content = resp.choices[0].message.content or ""
    return parse_recs(content)


# ---------------------------
# Main Form
# ---------------------------
col_left, col_right = st.columns([1.0, 0.9], gap="large")

with col_left:
    st.subheader("1) 입력 (Input Data)")
    with st.form("gift_form", clear_on_submit=False):
        relationship = st.selectbox("관계", ["친구", "연인", "가족", "선배", "후배", "기타"])
        closeness = st.selectbox("친밀도", ["어색", "보통", "친함"])
        occasion = st.selectbox("상황(이벤트)", ["생일", "기념일", "졸업", "입학", "감사", "기타"])
        budget = st.number_input("예산(원)", min_value=0, max_value=1_000_000, value=30000, step=1000)
        delivery = st.selectbox("전달 방식", ["직접", "택배", "급함(오늘/내일)"])
        taste = st.multiselect("취향 태그(복수 선택)", ["실용", "감성", "취미", "미니멀", "귀여움", "고급", "무난", "특별함"])
        keywords_like = st.text_input("좋아하는 것(키워드)", placeholder="예: 커피, 운동, 향, 심플한 스타일")
        keywords_avoid = st.text_input("피해야 할 것(키워드)", placeholder="예: 향 싫어함, 너무 비싼 건 부담, 알레르기 등")

        submitted = st.form_submit_button("🎁 추천 받기")

    sample = st.button("✨ 샘플 입력으로 채우기(선배/어색/졸업/3만원/커피)")
    if sample:
        st.session_state["_sample_fill"] = True

    if st.session_state.get("_sample_fill"):
        st.info("샘플 입력을 참고해서 직접 값만 맞춰서 다시 '추천 받기' 누르면 됩니다.")
        st.session_state["_sample_fill"] = False


with col_right:
    st.subheader("2) 위시리스트 (후보로 저장)")
    if len(st.session_state["wishlist"]) == 0:
        st.write("아직 저장한 후보가 없어요.")
    else:
        for i, rec in enumerate(st.session_state["wishlist"], start=1):
            with st.expander(f"{i}. {rec['title']} (점수 {rec['score']})"):
                st.write(rec.get("rationale", ""))
                st.write(f"검색어: `{rec.get('search_query','')}`")
                st.markdown(f"- 가격대: {rec.get('price_hint','')}")
                if st.button(f"🗑️ 삭제", key=f"wl_del_{i}"):
                    st.session_state["wishlist"].pop(i - 1)
                    st.rerun()


# ---------------------------
# Run recommendation
# ---------------------------
if submitted:
    if not openai_key:
        st.error("OpenAI API Key를 사이드바에 입력하세요.")
    elif not (naver_client_id and naver_client_secret):
        st.error("Naver Client ID / Secret을 사이드바에 입력하세요.")
    else:
        payload = {
            "relationship": relationship,
            "closeness": closeness,
            "occasion": occasion,
            "budget_krw": int(budget),
            "delivery": delivery,
            "taste_tags": taste,
            "likes": keywords_like,
            "avoids": keywords_avoid,
        }

        with st.spinner("추천 생성 중(OpenAI)…"):
            try:
                recs = get_recommendations(openai_key, payload)
            except Exception as e:
                st.exception(e)
                recs = []

        if recs:
            # For each recommendation, fetch real products via Naver Shopping Search API
            results_bundle = []
            with st.spinner("실제 판매 상품 검색 중(네이버 쇼핑)…"):
                for rec in recs:
                    query = rec.search_query
                    try:
                        api_json = naver_shop_search(
                            query=query,
                            client_id=naver_client_id,
                            client_secret=naver_client_secret,
                            display=30,
                            sort="sim",
                            exclude_used=True,
                        )
                        items = pick_best_items(api_json, budget=int(budget), top_k=3)
                    except Exception as e:
                        items = []
                    results_bundle.append((rec, items))

            st.session_state["last_result"] = {
                "payload": payload,
                "results_bundle": [
                    {
                        "rec": {
                            "rank": r.rank,
                            "title": r.title,
                            "score": r.score,
                            "rationale": r.rationale,
                            "price_hint": r.price_hint,
                            "search_query": r.search_query,
                            "alt_ideas": r.alt_ideas,
                        },
                        "items": items,
                    }
                    for (r, items) in results_bundle
                ],
            }


# ---------------------------
# Display results
# ---------------------------
if st.session_state["last_result"]:
    st.divider()
    st.subheader("3) 추천 결과 (순위 + 점수 근거 + 구매 링크 연결)")

    payload = st.session_state["last_result"]["payload"]
    bundle = st.session_state["last_result"]["results_bundle"]

    st.caption(
        f"입력 요약: 관계={payload['relationship']} / 친밀도={payload['closeness']} / 상황={payload['occasion']} / "
        f"예산={payload['budget_krw']:,}원 / 전달={payload['delivery']}"
    )

    for block in bundle:
        rec = block["rec"]
        items = block["items"]

        rank = rec["rank"]
        title = rec["title"]
        score = rec["score"]
        rationale = rec["rationale"]
        price_hint = rec["price_hint"]
        search_query = rec["search_query"]
        alt_ideas = rec["alt_ideas"]

        with st.container(border=True):
            top_cols = st.columns([0.15, 0.6, 0.25])
            with top_cols[0]:
                st.markdown(f"## #{rank}")
            with top_cols[1]:
                st.markdown(f"### {title}")
                st.write(f"**점수:** {score}  |  **가격대:** {price_hint}")
                st.write(f"**근거:** {rationale}")
                if alt_ideas:
                    st.write("**대체안:** " + " · ".join(alt_ideas[:3]))
            with top_cols[2]:
                # Action buttons
                decide_key = f"decide_{rank}_{title}"
                save_key = f"save_{rank}_{title}"
                fb_key = f"fb_{rank}_{title}"

                if st.button("✅ 이 선물로 결정", key=decide_key):
                    st.session_state["chosen"] = rec

                if st.button("💾 후보로 저장", key=save_key):
                    # prevent duplicates
                    exists = any(x.get("title") == title and x.get("rank") == rank for x in st.session_state["wishlist"])
                    if not exists:
                        st.session_state["wishlist"].append(rec)
                        st.toast("위시리스트에 저장했어요!", icon="💾")
                    else:
                        st.toast("이미 저장된 후보예요.", icon="ℹ️")

                st.link_button("🛒 바로 구매하기(검색)", make_naver_shop_search_url(search_query))

            # Real products section
            st.markdown("**실제 판매 상품(네이버 쇼핑 검색 결과)**")
            if not items:
                st.warning("예산 조건에 맞는 상품을 충분히 찾지 못했어요. 검색어를 바꾸거나 예산을 조정해보세요.")
                st.markdown(f"- 검색어: `{search_query}`")
            else:
                prod_cols = st.columns(3)
                for i, it in enumerate(items):
                    with prod_cols[i]:
                        if it.get("image"):
                            st.image(it["image"], use_container_width=True)
                        st.markdown(f"**{it['title']}**")
                        price = it.get("lprice", 0)
                        st.write(f"최저가: {price:,}원" if price else "가격 정보 없음")
                        if it.get("mallName"):
                            st.caption(f"판매처: {it['mallName']}")
                        if it.get("category"):
                            st.caption(it["category"])
                        if it.get("link"):
                            st.link_button("구매 링크", it["link"])

            # Decide flow details (purchase / price compare / nearby store)
            if st.session_state.get("chosen") and st.session_state["chosen"].get("title") == title and st.session_state["chosen"].get("rank") == rank:
                st.divider()
                st.markdown("### 🎯 결정 후 다음 행동(활용 흐름 연결)")
                decide_cols = st.columns(3)
                with decide_cols[0]:
                    st.link_button("🛒 구매 링크(네이버 쇼핑 검색)", make_naver_shop_search_url(search_query))
                with decide_cols[1]:
                    # "가격 비교"는 쇼핑 검색(여러 판매처) 자체가 비교 역할
                    st.link_button("💸 가격 비교(동일 검색)", make_naver_shop_search_url(search_query))
                with decide_cols[2]:
                    # 근처 매장 찾기: 네이버 지도 검색으로 연결
                    st.link_button("📍 근처 매장 찾기(지도)", make_naver_map_search_url(title))

                st.info("팁: 실제 서비스에선 '결정' 버튼 클릭 시 해당 추천의 대표 상품 1개를 자동 선택해 구매 링크를 바로 열 수도 있어요.")

            # Feedback section
            with st.expander("🧾 피드백 남기기(개인화 데이터)", expanded=False):
                st.write("이 추천이 실제로 도움이 됐는지 기록하면 다음 추천이 더 정확해집니다.")
                success = st.radio("결과", ["성공(만족)", "실패(별로)"], horizontal=True, key=f"succ_{rank}_{title}")
                stars = st.slider("별점", 1, 5, 4, key=f"stars_{rank}_{title}")
                memo = st.text_input("메모(선택)", key=f"memo_{rank}_{title}", placeholder="예: 너무 흔했음 / 취향에 딱 맞았음")
                if st.button("피드백 저장", key=f"savefb_{rank}_{title}"):
                    st.session_state["feedback"].append(
                        {
                            "title": title,
                            "rank": rank,
                            "score": score,
                            "success": success,
                            "stars": stars,
                            "memo": memo,
                            "search_query": search_query,
                            "input": payload,
                        }
                    )
                    st.toast("피드백을 저장했어요! (심화 단계에서 개인화에 활용)", icon="✅")

    st.divider()
    st.subheader("4) 저장된 피드백(개인화 데이터) 미리보기")
    if not st.session_state["feedback"]:
        st.write("아직 저장된 피드백이 없어요.")
    else:
        for i, fb in enumerate(st.session_state["feedback"], start=1):
            with st.expander(f"{i}. {fb['title']} | {fb['success']} | 별점 {fb['stars']}"):
                st.json(fb, expanded=False)


# ---------------------------
# Footer / Notes
# ---------------------------
st.divider()
st.caption(
    "참고: 네이버 쇼핑 검색 API는 검색어(query) 기반으로 결과를 반환하며, 정렬(sort)·상품 유형(filter/exclude) 등을 지원합니다. :contentReference[oaicite:6]{index=6}"
)
