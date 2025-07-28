#!/usr/bin/env python3
"""
Welcome Real Estate AI
=======================

This Streamlit application exposes a simple Hebrew chat interface for
exploring urban‑renewal data in Israel.  Users can ask questions such
as “כמה פרויקטים של התחדשות עירונית הוגשו בתל אביב ב‑2025?” and
receive data‑driven answers sourced from the Israeli government’s open
data portal (data.gov.il).  When a question includes a municipality
and a year, the app counts how many urban‑renewal complexes were
declared in that municipality during that year.  The underlying data
are retrieved on demand via the CKAN API and cached locally to avoid
unnecessary network traffic.

Important caveats:

* Only simple questions about counts of urban‑renewal complexes are
  supported in this pilot.  Questions that do not mention both a
  year and a municipality will receive a fallback answer.
* The app relies on being able to reach data.gov.il at runtime.  If
  you run this code in an environment without network access to
  data.gov.il, data loading will fail.  In such cases you may wish
  to supply a local CSV file instead of downloading from the API.
* The application is fully localized in Hebrew and intended for
  left‑to‑right layout use with Streamlit’s built‑in chat components.

To start the app locally, run:

    streamlit run app.py

"""

from __future__ import annotations

import os
import re
from typing import Optional

import pandas as pd
import requests
import streamlit as st


@st.cache_data(show_spinner=True)
def load_dataset(resource_id: str = "f65a0daf-f737-49c5-9424-d378d52104f5", *, limit: int = 32000) -> pd.DataFrame:
    """Load the urban renewal dataset from data.gov.il.

    The dataset describes urban‑renewal complexes announced under
    various government tracks.  Each record includes identifiers,
    municipality names, the number of existing and additional housing
    units, declaration dates, and other metadata.

    Args:
        resource_id: CKAN resource identifier for the dataset.
        limit: maximum number of records to retrieve.

    Returns:
        A pandas DataFrame containing all records in the dataset.

    Raises:
        requests.HTTPError: if the HTTP request fails.
        ValueError: if the response JSON is malformed.
    """
    api_url = "https://data.gov.il/api/3/action/datastore_search"
    params = {"resource_id": resource_id, "limit": limit}
    response = requests.get(api_url, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload.get("success"):
        raise ValueError("Failed to fetch dataset: success flag is false")
    records = payload["result"].get("records", [])
    df = pd.DataFrame.from_records(records)
    # Strip whitespace from textual columns and normalise missing values
    for col in ["Yeshuv", "ShemMitcham", "TaarichHachraza"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def extract_year(question: str) -> Optional[str]:
    """Extract a four‑digit year from the question.

    Searches for a sequence of four digits beginning with 19 or 20.
    Returns the first match if found; otherwise None.
    """
    match = re.search(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)", question)
    return match.group(0) if match else None


def extract_city(question: str, df: pd.DataFrame) -> Optional[str]:
    """Detect which municipality appears in the question.

    The function iterates over all unique municipality names in the
    dataset (found in the ``Yeshuv`` column) and returns the first
    municipality whose name is a substring of the question.  If no
    match is found, None is returned.

    Args:
        question: user query text.
        df: dataset containing the ``Yeshuv`` column.

    Returns:
        The matched municipality name, or None.
    """
    if "Yeshuv" not in df.columns:
        return None
    # Use a set for efficiency and drop NaN/empty strings
    for municipality in df["Yeshuv"].dropna().unique():
        name = str(municipality).strip()
        if not name:
            continue
        if name in question:
            return name
    return None


def answer_question(question: str, df: pd.DataFrame) -> str:
    """Generate an answer in Hebrew based on the question and dataset.

    Currently the function supports simple count queries of the form
    “כמה פרויקטים של התחדשות עירונית הוכרזו בעיר בשנת YYYY?”.  It
    detects a year and a municipality in the question; if both are
    present it counts how many complexes were declared in that
    municipality during that year using the ``TaarichHachraza`` and
    ``Yeshuv`` fields.  For other questions it returns a fallback
    message.

    Args:
        question: the user’s question in Hebrew.
        df: DataFrame containing urban‑renewal data.

    Returns:
        A Hebrew answer string.
    """
    question = question or ""
    # Extract year and city
    year = extract_year(question)
    city = extract_city(question, df)
    # For Tel‑Aviv the dataset uses "תל אביב יפו"; allow substring matching
    if city is None and "תל אביב" in question:
        city = "תל אביב יפו"
    if year and city:
        # Filter rows where declaration date includes the year and
        # municipality name matches.  Some rows may have missing
        # declaration dates, so dropna before filtering.
        mask = (
            df["Yeshuv"].str.contains(city, na=False)
            & df["TaarichHachraza"].str.contains(year, na=False)
        )
        subset = df.loc[mask]
        num_complexes = int(subset["MisparMitham"].nunique())
        # Compose the answer in Hebrew.  Use plural/singular correctly.
        if num_complexes == 0:
            return (
                f"לא נמצאו מתחמי התחדשות עירונית ב{city} בשנת {year}."
            )
        else:
            return (
                f"בשנת {year} הוכרזו {num_complexes} מתחמי התחדשות עירונית ב{city}."
            )
    # Fallback for unsupported queries
    return (
        "מצטער, כעת אני יכול לענות רק על שאלות הכוללות עיר ושנה. "
        "נסו למשל: 'כמה פרויקטים של התחדשות עירונית הוכרזו בתל אביב בשנת 2025?'"
    )


def main() -> None:
    """Run the Streamlit chat interface."""
    st.set_page_config(page_title="Welcome Real Estate AI", page_icon="🏠", layout="centered")
    st.title("Welcome Real Estate AI")
    # Introductory prompt
    st.write("תשאלו את Welcome Real Estate AI כל דבר שתרצו לדעת על נדל\"ן בישראל")
    # Load dataset (this may take a few seconds)
    try:
        df = load_dataset()
    except Exception as exc:
        st.error(f"שגיאה בטעינת הנתונים: {exc}")
        st.stop()
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # Display past messages
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        with st.chat_message(role):
            st.markdown(content)
    # Handle user input
    user_input = st.chat_input("הקלידו כאן את השאלה שלכם...")
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})
        # Generate answer
        answer = answer_question(user_input, df)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()