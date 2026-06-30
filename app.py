import streamlit as st
import os
import time
from dotenv import load_dotenv

from graph import build_graph

load_dotenv()

st.set_page_config(page_title="Academic Research Co-Pilot", layout="wide")

st.title("📚 Academic Research Co-Pilot")
st.markdown("Enter a topic or hypothesis, and the agent will query arXiv/PubMed, extract methodologies and datasets, and generate a formatted bibliography.")

# Check for API key
if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
    st.warning("Please set your `GROQ_API_KEY` in the `.env` file to proceed.")
else:
    topic = st.text_input("Enter Topic or Hypothesis:")
    
    if st.button("Run Research Workflow"):
        if topic:
            with st.spinner("Initializing Agents..."):
                app_graph = build_graph()
                
            st.info("Agents are running... Please wait. This may take a minute depending on the queries.")
            
            initial_state = {"topic": topic}
            
            try:
                start_time = time.time()
                final_state = app_graph.invoke(initial_state)
                end_time = time.time()
                latency = end_time - start_time
                st.success(f"Research Complete! (Took {latency:.2f} seconds)")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Agent Operations")
                    st.write(f"**Selected Strategy**: {final_state.get('search_strategy')}")
                    st.write("**Search Queries used**:")
                    for q in final_state.get("search_queries", []):
                        st.write(f"- {q}")
                        
                with col2:
                    st.subheader("Bibliography & Analysis")
                    st.markdown(final_state.get("bibliography", ""))
                    
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a topic.")
