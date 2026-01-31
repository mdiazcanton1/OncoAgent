"""Streamlit UI for OncoAgent."""

from __future__ import annotations

import base64

import requests
import streamlit as st


st.set_page_config(page_title="OncoAgent", layout="wide")
st.title("OncoAgent - Clinical Research Assistant")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

with st.sidebar:
    api_base = st.text_input("API Base URL", value="http://localhost:8000")
    if st.button("New conversation"):
        st.session_state.thread_id = None
        st.session_state.chat_history = []

if st.session_state.chat_history:
    st.subheader("Conversation")
    for role, content in st.session_state.chat_history:
        st.markdown(f"**{role}:** {content}")

query = st.text_area("Clinical question", height=120)
cancer_type = st.text_input("Cancer type (optional)")
uploaded_files = st.file_uploader(
    "Upload pathology scans or charts (optional)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)

if st.button("Run"):
    if not query.strip():
        st.warning("Please enter a clinical question.")
    else:
        images_payload = []
        for file in uploaded_files or []:
            data = base64.b64encode(file.read()).decode("utf-8")
            images_payload.append(f"data:{file.type};base64,{data}")

        st.session_state.chat_history.append(("User", query))

        payload = {
            "query": query,
            "thread_id": st.session_state.thread_id,
            "cancer_type": cancer_type or None,
            "images": images_payload or None,
        }

        with st.spinner("Querying OncoAgent..."):
            response = requests.post(f"{api_base}/query", json=payload, timeout=120)

        if response.status_code != 200:
            st.error(f"API error: {response.status_code} - {response.text}")
        else:
            data = response.json()
            st.session_state.thread_id = data.get("thread_id", st.session_state.thread_id)
            st.session_state.chat_history.append(("Assistant", data.get("response", "")))
            st.subheader("Response")
            st.write(data.get("response", ""))
            st.subheader("Confidence")
            st.write(data.get("confidence", "UNCERTAIN"))

            if data.get("clinical_trials"):
                st.subheader("Clinical Trials")
                st.write(data["clinical_trials"])

            if data.get("citations"):
                st.subheader("Citations")
                st.write(data["citations"])

