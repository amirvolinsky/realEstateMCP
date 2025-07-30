#!/usr/bin/env python3
import os
import re
from typing import Optional

import pandas as pd
import requests
import streamlit as st


@st.cache_data(show_spinner=True)
def load_dataset_mcp(dataset_name="urban_renewal", limit=32000) -> pd.DataFrame:
    """Load dataset via the MCP server."""
    mcp_url = os.environ.get("DATAGOV_MCP_URL", "https://datagov-mcp.onrender.com")
    api_url = f"{mcp_url}/fetch_data"
    params = {"dataset_name": dataset_name, "limit": limit}
    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()
    records = response.json()
    df = pd.DataFrame.from_records(records)
    for col in ["Yeshuv", "ShemMitcham", "TaarichHachraza"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def extract_year(question: str) -> Optional[str]:
    match = re.search(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)", question)
    return match.group(0) if match else None


def extract_city(question: str, df: pd.DataFrame) -> Optional[str]:
    if "Yeshuv" not in df.columns:
        return None
    for municipality in df["Yeshuv"].dropna().unique():
        name = str(municipality).strip()
        if not name:
            continue
        if name in question:
            return name
    return None


def answer_question(question: str, df: pd.DataFrame) -> str:
    question = question or ""
    year = extract_year(question)
    city = extract_city(question, df)
    if city is None and "תל אביב" in question:
        city = "תל אביב יפו"
    if year and city:
        mask = (
            df["Yeshuv"].str.contains(city, na=False)
            & df["TaarichHachraza"].str.contains(year, na=False)
        )
        subset = df.loc[mask]
        num_complexes = int(subset["MisparMitham"].nunique())
        if num_complexes == 0:
            return f"לא נמצאו מתחמי התחדשות עירונית ב{city} בשנת {year}."
        else:
            return f"בשנת {year} הוכרזו {num_complexes} מתחמי התחדשות עירונית ב{city}."
    return (
        "מצטער, כעת אני יכול לענות רק על שאלות הכוללות עיר ושנה. "
        "נסו למשל: 'כמה פרויקטים של התחדשות עירונית הוכרזו בתל אביב בשנת 2025?'"
    )


def main() -> None:
    st.set_page_config(page_title="Welcome Real Estate AI", page_icon="🏠", layout="centered")
    st.title("Welcome Real Estate AI")
    st.write("תשאלו את Welcome Real Estate AI כל דבר שתרצו לדעת על נדל"ן בישראל")

    try:
        df = load_dataset_mcp()
    except Exception as exc:
        st.error(f"שגיאה בטעינת הנתונים: {exc}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)

    user_input = st.chat_input("הקלידו כאן את השאלה שלכם...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        answer = answer_question(user_input, df)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
