import arxiv
from Bio import Entrez
import datetime

# Always set an email for Entrez to avoid being blocked
Entrez.email = "reesh9805@gmail.com" 

def search_arxiv(query: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for papers and return metadata."""
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    results = []
    try:
        for paper in client.results(search):
            results.append({
                "source": "arxiv",
                "title": paper.title,
                "authors": [author.name for author in paper.authors],
                "date": paper.published.strftime("%Y-%m-%d") if paper.published else "",
                "abstract": paper.summary,
                "url": paper.pdf_url or paper.entry_id
            })
    except Exception as e:
        print(f"Error fetching from arxiv: {e}")
    return results

def search_pubmed(query: str, max_results: int = 5) -> list[dict]:
    """Search PubMed for papers and return metadata."""
    results = []
    try:
        # Step 1: Search to get IDs
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="date")
        record = Entrez.read(handle)
        handle.close()
        
        id_list = record.get("IdList", [])
        if not id_list:
            return results
            
        # Step 2: Fetch details for those IDs
        fetch_handle = Entrez.efetch(db="pubmed", id=",".join(id_list), retmode="xml")
        fetch_records = Entrez.read(fetch_handle)
        fetch_handle.close()
        
        articles = fetch_records.get('PubmedArticle', [])
        
        for article in articles:
            medline = article.get('MedlineCitation', {})
            article_info = medline.get('Article', {})
            
            title = article_info.get('ArticleTitle', '')
            abstract_texts = article_info.get('Abstract', {}).get('AbstractText', [])
            abstract = " ".join([str(text) for text in abstract_texts]) if abstract_texts else ""
            
            # Extract authors
            author_list = article_info.get('AuthorList', [])
            authors = []
            for author in author_list:
                last_name = author.get('LastName', '')
                fore_name = author.get('ForeName', '')
                authors.append(f"{fore_name} {last_name}".strip())
                
            # Date
            pub_date = article_info.get('Journal', {}).get('JournalIssue', {}).get('PubDate', {})
            year = pub_date.get('Year', '')
            month = pub_date.get('Month', '')
            date = f"{year} {month}".strip()
            
            pmid = medline.get('PMID', '')
            url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""
            
            results.append({
                "source": "pubmed",
                "title": title,
                "authors": authors,
                "date": date,
                "abstract": abstract,
                "url": url
            })
            
    except Exception as e:
        print(f"Error fetching from pubmed: {e}")
        
    return results
