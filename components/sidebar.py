import streamlit as st

from components.faq import faq

def sidebar():
    with st.sidebar:
        # st.markdown(
        #     "## How to Use VCPilot\n"
        #     "1. Input the URL or upload articles, blogs, or documents related to an AI startup.📄\n"
        #     "2. Review the automatically generated risk profile and potential analysis by VCPilot.🔍\n"
        #     "3. Explore detailed insights and follow-up questions to further understand the startup's viability and innovation.💡\n"
        # )

        # Assuming VC Pilot requires an API key for functionality
        # vc_pilot_api_key = st.text_input(
        #     "OpenAI API Key",
        #     type="password",
        #     placeholder="Paste your OpenAI API key here",
        #     help="You can obtain your API key from the VC Pilot dashboard.",
        #     value=st.secrets.openai.key
        #     or st.session_state.get("OPEN_API_KEY", ""),
        # )

        # st.session_state["OPEN_API_KEY"] = vc_pilot_api_key

        st.markdown("---")
        st.markdown("# About VCPilot")
        st.markdown(
            "VCPilot assists venture capital investors and analysts in evaluating AI technology startups by analyzing articles, websites, and blogs, providing a focused lens for swift assessment of their credibility, innovation, and market fit."
        )
        st.markdown(
            "Our AI-driven approach offers relevant and actionable insights, streamlining the task of identifying potential leaders in the AI revolution."
        )
        st.markdown(
            "Developed by a team dedicated to enhancing venture capital investment strategies, VCPilot is your go-to tool for navigating the dynamic AI startup landscape."
        )
        st.markdown("---")

        faq()