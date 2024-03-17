import streamlit as st
import anthropic
from components.sidebar import sidebar

sidebar()

client = anthropic.Client(api_key=st.secrets.anthropic.key)

st.title("VC pilot Claude-3-opus")

if question := st.chat_input("How risky is this project?:"):
    st.chat_message("user").markdown(question)
    prompt = f"""{anthropic.HUMAN_PROMPT} {question} {anthropic.AI_PROMPT}"""
    print(prompt)


    response = client.messages.create(
        model="claude-3-opus-20240229",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": question}
        ]
    )
    st.write("### Answer")
    st.write(response.content[0].text)