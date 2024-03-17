import streamlit as st

from components.faq import faq

def sidebar():
    with st.sidebar:
        st.image('./assets/logo.png', width=300)
        st.markdown("---")
        st.markdown("# About VC Pilot")
        st.markdown(
            "VC Pilot assists venture capital investors and analysts in evaluating AI technology startups by analyzing articles, websites, and blogs, providing a focused lens for swift assessment of their credibility, innovation, and market fit."
        )
        st.markdown(
            "Our AI-driven approach offers relevant and actionable insights, streamlining the task of identifying potential leaders in the AI revolution."
        )
        st.markdown(
            "Developed by a team dedicated to enhancing venture capital investment strategies, VC Pilot is your go-to tool for navigating the dynamic AI startup landscape."
        )
        st.markdown("---")

        faq()