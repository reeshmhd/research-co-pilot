import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from tools import search_arxiv, search_pubmed

class AgentState(TypedDict):
    topic: str
    search_strategy: str
    search_queries: List[str]
    raw_papers: List[Dict]
    summaries: List[Dict]
    bibliography: str

class SearchStrategy(BaseModel):
    strategy: str = Field(description="The database to search: 'arxiv', 'pubmed', or 'both'")
    queries: List[str] = Field(description="List of optimized search queries.")

class ExtractionSummary(BaseModel):
    methodology: str = Field(description="Summary of the methodology used in the paper.")
    datasets: str = Field(description="Datasets explicitly mentioned in the paper. If none, state 'No explicit datasets mentioned.'")

def orchestrator_node(state: AgentState):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Analyze the topic and decide if we should search 'arxiv' (physics, CS, math), 'pubmed' (medicine, biology), or 'both'. Then generate 1 or 2 highly optimized search queries."),
        ("user", "Topic: {topic}")
    ])
    
    chain = prompt | llm.with_structured_output(SearchStrategy)
    result = chain.invoke({"topic": state["topic"]})
    
    return {"search_strategy": result.strategy, "search_queries": result.queries}

def retriever_node(state: AgentState):
    strategy = state.get("search_strategy", "both").lower()
    queries = state.get("search_queries", [])
    
    papers = []
    
    # Try the first query for simplicity
    if queries:
        query = queries[0]
    else:
        query = state["topic"]
    
    if "arxiv" in strategy or "both" in strategy:
        arxiv_results = search_arxiv(query, max_results=5)
        papers.extend(arxiv_results)
        
    if "pubmed" in strategy or "both" in strategy:
        limit = 3 if "both" in strategy else 5
        pubmed_results = search_pubmed(query, max_results=limit)
        papers.extend(pubmed_results)
        
    papers = papers[:5]
    
    return {"raw_papers": papers}

def summarizer_node(state: AgentState):
    papers = state.get("raw_papers", [])
    if not papers:
        return {"summaries": []}
        
    llm = ChatGroq(model="llama3-70b-8192", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert methodology extractor. Given an abstract, concisely summarize the methodology used and extract any mentioned datasets."),
        ("user", "Abstract: {abstract}")
    ])
    
    chain = prompt | llm.with_structured_output(ExtractionSummary)
    
    summaries = []
    for paper in papers:
        if paper.get("abstract"):
            try:
                res = chain.invoke({"abstract": paper["abstract"]})
                summary_data = {
                    "methodology": res.methodology,
                    "datasets": res.datasets,
                    "paper_title": paper["title"]
                }
                summaries.append(summary_data)
            except Exception as e:
                print(f"Extraction failed for {paper['title']}: {e}")
                summaries.append({
                    "methodology": "Failed to extract methodology due to error.",
                    "datasets": "Failed to extract datasets due to error.",
                    "paper_title": paper["title"]
                })
        else:
            summaries.append({
                "methodology": "No abstract available to extract methodology.",
                "datasets": "No abstract available.",
                "paper_title": paper["title"]
            })
            
    return {"summaries": summaries}

def bibliography_node(state: AgentState):
    papers = state.get("raw_papers", [])
    summaries = state.get("summaries", [])
    
    if not papers:
        return {"bibliography": "No papers found for the given topic."}
        
    bibliography = "## Annotated Bibliography\n\n"
    
    for i, paper in enumerate(papers):
        title = paper.get("title", "Untitled")
        authors = ", ".join(paper.get("authors", []))
        date = paper.get("date", "n.d.")
        url = paper.get("url", "#")
        source = paper.get("source", "Unknown").capitalize()
        
        summary = next((s for s in summaries if s["paper_title"] == title), None)
        methodology = summary["methodology"] if summary else "N/A"
        datasets = summary["datasets"] if summary else "N/A"
        
        bibliography += f"### {i+1}. [{title}]({url})\n"
        bibliography += f"- **Authors**: {authors}\n"
        bibliography += f"- **Date**: {date} ({source})\n"
        bibliography += f"- **Methodology**: {methodology}\n"
        bibliography += f"- **Datasets**: {datasets}\n\n"
        
    return {"bibliography": bibliography}

def build_graph():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("summarizer", summarizer_node)
    workflow.add_node("bibliography", bibliography_node)
    
    workflow.set_entry_point("orchestrator")
    workflow.add_edge("orchestrator", "retriever")
    workflow.add_edge("retriever", "summarizer")
    workflow.add_edge("retriever", "bibliography")
    workflow.add_edge("summarizer", END)
    workflow.add_edge("bibliography", END)
    
    return workflow.compile()
