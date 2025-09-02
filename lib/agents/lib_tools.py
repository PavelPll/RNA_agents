# Adapted from code available at: https://github.com/techwithtim/PythonAIAgentFromScratch/blob/main/tools.py
# Tools are: 
# - save_tool: save output results for each evolution step to file
# - search_tool: search in internet using DuckDuckGo
# - wiki_tool
# - semantic_scholar_tool: semanticscholar (scientific articles)
# - europe_pmc_tool: PMCSearch (scientific articles)
# - pubmed_search_tool or biopython_pubmed_tool: PubMedSearch (scientific articles)

from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import requests

def save_to_txt(data: str, filename: str = "../data/processed/rna_evolution/evolution_steps.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

############################################################
class SemanticScholarSearch:
    def run(self, query: str) -> str:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": 3,
            "fields": "title,authors,url,abstract,year"
        }
        response = requests.get(url, params=params)
        results = response.json().get("data", [])

        if not results:
            return "No results found."

        output = ""
        for i, paper in enumerate(results, 1):
            output += f"{i}. {paper.get('title')} ({paper.get('year')})\n"
            if paper.get("abstract"):
                output += f"Abstract: {paper['abstract'][:300]}...\n"
            output += f"Link: {paper.get('url')}\n\n"
        return output.strip()

# LangChain-compatible tool
semantic_scholar_search = SemanticScholarSearch()
semantic_scholar_tool = Tool(
    name="semantic_scholar",
    func=semantic_scholar_search.run,
    description="Search for academic papers on RNA, evolution, and biology."
)

class EuropePMCSearch:
    def run(self, query: str) -> str:
        url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
        params = {
            "query": query,
            "format": "json",
            "pageSize": 3,
        }
        response = requests.get(url, params=params)
        results = response.json().get("resultList", {}).get("result", [])

        if not results:
            return "No results found."

        output = ""
        for i, article in enumerate(results, 1):
            output += f"{i}. {article.get('title')} ({article.get('pubYear')})\n"
            output += f"Journal: {article.get('journalTitle')}\n"
            output += f"Link: https://europepmc.org/article/{article.get('source')}/{article.get('id')}\n\n"
        return output.strip()

# LangChain-compatible tool
europe_pmc_search = EuropePMCSearch()
europe_pmc_tool = Tool(
    name="europe_pmc",
    func=europe_pmc_search.run,
    description="Search biomedical literature for RNA evolution research using Europe PMC."
)

from easy_entrez import EntrezAPI
#from langchain_core.tools import Tool

class PubMedSearch:
    def __init__(self, email: str, tool_name: str = "my_tool", api_key: str | None = None):
        self.client = EntrezAPI(tool_name, email, return_type="json", api_key=api_key)
    
    def run(self, query: str) -> str:
        result = self.client.search(query, database="pubmed", max_results=5)
        ids = result.data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return "No results found."
        detail = self.client.fetch(ids, database="pubmed", max_results=5)
        # Quick parsing; real world would parse JSON/XML properly
        return f"Found PMIDs: {', '.join(ids)} — Check details in fetched data."

pubmed_search_tool = Tool(
    name="pubmed_search",
    func=PubMedSearch(email="your-email@example.com", api_key=None).run,
    description="Search PubMed articles via Entrez API."
)

# Alternative using Biopython's Bio.Entrez
from Bio import Entrez
Entrez.email = 'pp60008@gmail.com'
#Entrez.email = "your-email@example.com"  # Required per NCBI policy
# Optional: Entrez.api_key = "YOUR_API_KEY"

class BiopythonPubMedSearch:
    def run(self, query: str) -> str:
        search = Entrez.esearch(db="pubmed", term=query, retmax=4)
        ids = Entrez.read(search)["IdList"]
        if not ids:
            return "No results found."
        fetch = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
        abstracts = fetch.read().split("\n\n")
        return "\n\n".join(f"{i+1}. {abstracts[i][:500]}" for i in range(len(abstracts)))

biopython_pubmed_tool = Tool(
    name="biopython_pubmed",
    func=BiopythonPubMedSearch().run,
    description="Search and fetch PubMed abstracts using Biopython Entrez."
)