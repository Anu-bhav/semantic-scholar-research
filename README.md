# Semantic Scholar Research Tool

This is a command-line tool that interacts with the [Semantic Scholar](https://www.semanticscholar.org/) API to retrieve information about academic papers and authors. It allows you to:

- Fetch details for specific papers using various identifiers (DOI, Semantic Scholar ID, arXiv ID, etc.).
- Retrieve information about authors, including their publications and citation metrics.
- Search for papers based on keywords, titles, or authors.
- Download PDFs of papers (when available).
- Get paper recommendations.

The tool handles API rate limits, includes error handling, and supports parallel PDF downloads for efficiency. It also features in-memory caching to reduce API calls.

## Prerequisites

- Python 3.7+
- Required packages: `httpx`, `aiocache`, `colorama`, `parsel`, `tenacity`. Install them using pip:

  ```bash
  uv venv
  uv pip install -r requirements.txt
  ```

## Installation

1.  Clone this repository or download the script (`semantic-scholar-research.py`).
2.  (Optional, but highly recommended) Obtain a Semantic Scholar API key from [https://www.semanticscholar.org/developer/register](https://www.semanticscholar.org/developer/register). An API key provides higher rate limits.

## Usage

python semantic-scholar-research.py <type> <id> [options]

### Arguments

- `<type>` (Required): The type of operation to perform. Choose one of:

  - `paper`: Retrieve details about a specific paper.
  - `author`: Retrieve details about a specific author.
  - `search`: Perform a search for papers or authors.

- `<id>` (Required): The identifier for the operation.
  - For `paper` type: A paper identifier (see "Paper ID Formats" below).
  - For `author` type: A Semantic Scholar Author ID (e.g., `1741101`).
  - For `search` type: The search query string (e.g., `"quantum computing"`).

### Options

- **`-s`**, **`--search_type`** (Only for `search` type): Specifies the type of search. Defaults to `relevance`.

  - `relevance`: General keyword search, sorted by relevance.
  - `title`: Search for a paper by its exact title.
  - `bulk`: Bulk search for papers (for larger result sets).
  - `author`: Search for authors by name.

- **`-d`**, **`--detail_level`**: Controls the amount of information retrieved. Defaults to `basic`.

  - `basic`: Essential information (title, abstract, year, authors, URL, external IDs).
  - `detailed`: Includes basic details plus references, citations, venue, and influential citation count.
  - `complete`: The most comprehensive data (all fields from `detailed` plus publication venue details, fields of study, etc.).

- **`-dl`**, **`--download`**: Attempts to download PDFs for retrieved papers (if available).

- **`-l`**, **`--limit`** (For `search` and `author` types): Maximum number of results to return. Defaults to 5. For `search`, this affects the initial search; for `author`, it limits the number of papers listed.

- **`-so`**, **`--sort_by`** (For `search` type when using `--search_type bulk`): Sorts bulk search results. Defaults to `year-desc`.

  - `year`: Publication year, oldest first.
  - `year-desc`: Publication year, newest first.
  - `citationCount`: Citation count, most cited first.
  - `citationCount-asc`: Citation count, least cited first.
  - `paperId`: Semantic Scholar Paper ID, ascending.
  - `paperId-desc`: Semantic Scholar Paper ID, descending.

- **`-h`**, **`--help`**: Displays the help message.

### Paper ID Formats

The `paper` type accepts the following identifier formats:

- Semantic Scholar ID: e.g., `649def34f8be52c8b66281af98ae884c09aef38b`
- CorpusId: e.g., `CorpusId:215416146`
- DOI: e.g., `DOI:10.18653/v1/N18-3011`
- ARXIV: e.g., `ARXIV:2106.15928`
- MAG: e.g., `MAG:112218234`
- ACL: e.g., `ACL:W12-3903`
- PMID: e.g., `PMID:19872477`
- PMCID: e.g., `PMCID:2323736`
- URL: e.g., `URL:https://arxiv.org/abs/2106.15928v1` (Supported domains: semanticscholar.org, arxiv.org, aclweb.org, acm.org, biorxiv.org)

### Environment Variables

- **`SEMANTIC_SCHOLAR_API_KEY`** (Optional, but recommended): Your Semantic Scholar API key. Set this environment variable to use your key. This gives you higher rate limits and access to additional features. Without a key, the script uses unauthenticated access (with lower rate limits).

  - **Linux/macOS:**

    ```bash
    export SEMANTIC_SCHOLAR_API_KEY="your_api_key_here"
    ```

  - **Windows (PowerShell):**

    ```powershell
    $env:SEMANTIC_SCHOLAR_API_KEY="your_api_key_here"
    ```

  - **Windows (cmd):**

    ```cmd
    set SEMANTIC_SCHOLAR_API_KEY=your_api_key_here
    ```

### Examples

1.  **Get basic details for a paper by DOI:**

    ```
    python semantic-scholar-research.py paper DOI:10.18653/v1/N18-3011
    ```

2.  **Get complete details for a paper by Semantic Scholar ID:**

    ```
    python semantic-scholar-research.py paper 649def34f8be52c8b66281af98ae884c09aef38b -d complete
    ```

3.  **Get basic details for an author:**

    ```
    python semantic-scholar-research.py author 1741101
    ```

4.  **Search for papers related to "quantum computing" (relevance search):**

    ```
    python semantic-scholar-research.py search "quantum computing"
    ```

5.  **Search for a paper by its exact title:**

    ```
    python semantic-scholar-research.py search "Attention is all you need" -s title
    ```

6.  **Perform a bulk search for "machine learning", sorting by year (oldest first):**

    ```
    python semantic-scholar-research.py search "machine learning" -s bulk -so year
    ```

7.  **Search for papers related to "deep learning" and download PDFs:**

    ```
    python semantic-scholar-research.py search "deep learning" -dl -l 10
    ```

8.  **Search for authors named "Yoshua Bengio":**

    ```
    python semantic-scholar-research.py search "Yoshua Bengio" -s author
    ```

9.  **Get detailed information about an author and list their top 10 papers:**

    ```
    python semantic-scholar-research.py author 1741101 -d detailed -l 10
    ```

10. **Get paper recommendations for a paper by DOI:**

    ```
    python semantic-scholar-research.py paper DOI:10.1038/s41586-021-03464-x -d basic -l 5
    ```

11. **Get paper recommendations, considering multiple papers:**
    ```
     # Assuming a script to get multiple recommendations is implemented
     python semantic-scholar-research.py recommendations "DOI:10.1038/s41586-021-03464-x,DOI:10.1126/science.1241480" -l 5
    ```

## Notes

- **Rate Limits:** The Semantic Scholar API has rate limits. Using an API key significantly increases these limits. The script includes a rate limiter that automatically slows down requests if necessary.
- **PDF Downloads:** PDF downloads are attempted only if a DOI is available and a direct PDF link can be found (either through the API or by scraping the paper's webpage). Success is not guaranteed, as it depends on the publisher and website structure.
- **Error Handling:** The script includes error handling for common issues like API errors, timeouts, and invalid input. Informative error messages are displayed.
- **Caching**: The script uses in-memory caching. The cache is cleared when you restart the script.
- **Recommendations:** The script can retrieve paper recommendations using the `/recommendations/v1/papers/forpaper/{paper_id}` and `/recommendations/v1/papers` endpoints.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues on the GitHub repository.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project is inspired by and incorporates parts of the code from [semantic-scholar-fastmcp-mcp-server](https://github.com/YUZongmin/semantic-scholar-fastmcp-mcp-server) by YUZongmin, which was originally designed for an MCP server. This project significantly modifies and extends the original code to create a standalone command-line research tool with additional features. We thank the original author for their contributions to the open-source community.
