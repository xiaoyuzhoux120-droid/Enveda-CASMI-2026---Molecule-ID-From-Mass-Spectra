"""Literature search over the official, free public APIs of arXiv, Semantic Scholar and Crossref.

  .venv/bin/python scripts/lit_search.py "MS/MS spectra molecular structure retrieval" -n 5
  .venv/bin/python scripts/lit_search.py "MassSpecGym" --source s2

No API keys are required. If SEMANTIC_SCHOLAR_API_KEY is set (optional, free key),
it is sent as the x-api-key header for higher rate limits; it is never printed.
"""
import argparse
import os
import time
import xml.etree.ElementTree as ET

import requests

UA = {"User-Agent": "casmi-research-lit-search/1.0"}


def arxiv(q, n):
    r = requests.get("https://export.arxiv.org/api/query", headers=UA, timeout=30,
                     params={"search_query": " AND ".join(f"all:{w}" for w in q.split()),
                             "max_results": n, "sortBy": "relevance"})
    r.raise_for_status()
    ns = {"a": "http://www.w3.org/2005/Atom"}
    for e in ET.fromstring(r.text).findall("a:entry", ns):
        yield {"title": " ".join(e.find("a:title", ns).text.split()),
               "year": e.find("a:published", ns).text[:4],
               "url": e.find("a:id", ns).text}


def s2(q, n):
    headers = dict(UA)
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    for attempt in range(4):  # unauthenticated requests share a global rate limit
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", headers=headers,
                         timeout=30, params={"query": q, "limit": n,
                                             "fields": "title,year,citationCount,externalIds,url"})
        if r.status_code != 429:
            break
        time.sleep(2 ** attempt * 2)
    r.raise_for_status()
    for p in r.json().get("data", []):
        doi = (p.get("externalIds") or {}).get("DOI")
        yield {"title": p["title"], "year": p.get("year"), "citations": p.get("citationCount"),
               "url": f"https://doi.org/{doi}" if doi else p.get("url")}


def crossref(q, n):
    r = requests.get("https://api.crossref.org/works", headers=UA, timeout=30,
                     params={"query.bibliographic": q, "rows": n,
                             "select": "title,DOI,issued,is-referenced-by-count"})
    r.raise_for_status()
    for it in r.json()["message"]["items"]:
        yield {"title": (it.get("title") or [""])[0],
               "year": (it.get("issued", {}).get("date-parts") or [[None]])[0][0],
               "citations": it.get("is-referenced-by-count"), "url": f"https://doi.org/{it['DOI']}"}


SOURCES = {"arxiv": arxiv, "s2": s2, "crossref": crossref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--source", choices=list(SOURCES) + ["all"], default="all")
    args = ap.parse_args()
    for name in (SOURCES if args.source == "all" else [args.source]):
        print(f"## {name}")
        try:
            for hit in SOURCES[name](args.query, args.n):
                print(" -", " | ".join(str(v) for v in hit.values() if v is not None))
        except requests.HTTPError as exc:
            print("  error:", exc.response.status_code, exc.response.reason)


if __name__ == "__main__":
    main()
