import streamlit as st
from openai import OpenAI

st.title("🤖 나의 AI 챗봇")

# 사이드바에서 API Key 입력
api_key = st.sidebar.text_input("OpenAI API Key", type="password")
mood = st.sidebar.selectbox(
    "오늘의 기분을 선택하세요",
    ["행복", "보통", "피곤", "스트레스", "슬픔", "설렘"],
)
st.sidebar.caption(f"선택한 기분: {mood}")

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("메시지를 입력하세요"):
    if not api_key:
        st.error("⚠️ 사이드바에서 API Key를 입력해주세요!")
    else:
        # 사용자 메시지 저장 및 표시
        st.session_state.messages.append(
            {"role": "user", "content": f"[기분: {mood}] {prompt}"}
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 응답 생성
        with st.chat_message("assistant"):
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages
            )
            reply = response.choices[0].message.content
            st.markdown(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply})
