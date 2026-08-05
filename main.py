import streamlit as st
from backend import (
    build_and_save_index,
    answer_question,
    clean_urls,
    index_exists,
)

st.set_page_config(page_title="Ask-The-URL", page_icon="assets/icons8-ask-32.png")

st.title("Ask-The-Url")
st.sidebar.title("Enter Urls")

urls = [st.sidebar.text_input(f"Url {i + 1}") for i in range(3)]

process_url_clicked = st.sidebar.button("Process Urls")

status_placeholder = st.empty()

if process_url_clicked:
    cleaned_urls = clean_urls(urls)

    if not cleaned_urls:
        st.sidebar.error("Please enter at least one valid Url before processing.")
    else:
        try:
            build_and_save_index(cleaned_urls, status_callback=status_placeholder.text)
            status_placeholder.success("Done! You can ask a question below.")
        except Exception as exc:
            status_placeholder.error(f"Something went wrong while processing the Urls: {exc}")

query = st.text_input("Ask Your Question About the Urls")

if query:
    if index_exists():
        try:
            result = answer_question(query)

            st.header("Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources: ")
                for source in sources.split("\n"):
                    if source.strip():
                        st.write(source)
        except Exception as exc:
            st.error(f"Something went wrong while answering: {exc}")
    else:
        st.warning("Please process some Urls first.")
