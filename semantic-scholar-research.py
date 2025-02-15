#!/usr/bin/env python3
import argparse
import asyncio
import logging
import os
import re
import sys
import time
import traceback
from enum import Enum
from typing import Dict, List, Optional, Tuple

import colorama
import httpx
from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer
from colorama import Fore, Style
from parsel import Selector
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

colorama.init(autoreset=True)  # Initialize colorama and enable auto-reset


logging.basicConfig(level=logging.CRITICAL)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Global HTTP client for connection pooling
http_client = None

SEMANTIC_SCHOLAR_API_KEY = ""

# --- Caching (Global) ---
paper_cache = {}  # Simple in-memory cache: {paper_id: data}
author_cache = {}  # Simple in-memory cache: {author_id: data}
search_cache = {}


# Error Types
class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    API_ERROR = "api_error"
    VALIDATION = "validation"
    TIMEOUT = "timeout"


# --- Field Constants ---
class PaperFields:
    DEFAULT = ["title", "abstract", "year", "citationCount", "authors", "url", "externalIds"]
    DETAILED = DEFAULT + [
        "references",
        "citations",
        "venue",
        "influentialCitationCount",
    ]
    MINIMAL = ["title", "year", "authors"]
    SEARCH = ["paperId", "title", "year", "citationCount"]
    BULK_SEARCH = [
        "paperId",
        "title",
        "year",
        "citationCount",
        "abstract",
        "venue",
        "authors",
        "externalIds",
    ]

    # Valid fields from API documentation
    VALID_FIELDS = {
        "abstract",
        "authors",
        "citationCount",
        "citations",
        "corpusId",
        "embedding",
        "externalIds",
        "fieldsOfStudy",
        "influentialCitationCount",
        "isOpenAccess",
        "openAccessPdf",
        "paperId",
        "publicationDate",
        "publicationTypes",
        "publicationVenue",
        "references",
        "s2FieldsOfStudy",
        "title",
        "tldr",
        "url",
        "venue",
        "year",
    }


class AuthorDetailFields:
    """Common field combinations for author details"""

    # Basic author information
    BASIC = ["name", "url", "affiliations"]

    # Author's papers information
    PAPERS_BASIC = ["papers"]  # Returns paperId and title
    PAPERS_DETAILED = [
        "papers.year",
        "papers.authors",
        "papers.abstract",
        "papers.venue",
        "papers.url",
    ]

    # Complete author profile
    COMPLETE = BASIC + PAPERS_BASIC + PAPERS_DETAILED

    # Citation metrics
    METRICS = ["citationCount", "hIndex", "paperCount"]

    # Valid fields for author details
    VALID_FIELDS = {
        "authorId",
        "name",
        "url",
        "affiliations",
        "papers",
        "papers.year",
        "papers.authors",
        "papers.abstract",
        "papers.venue",
        "papers.url",
        "citationCount",
        "hIndex",
        "paperCount",
    }


class PaperDetailFields:
    """Common field combinations for paper details"""

    # Basic paper information
    BASIC = ["title", "abstract", "year", "venue"]

    # Author information
    AUTHOR_BASIC = ["authors"]
    AUTHOR_DETAILED = ["authors.url", "authors.paperCount", "authors.citationCount"]

    # Citation information
    CITATION_BASIC = ["citations", "references"]
    CITATION_DETAILED = [
        "citations.title",
        "citations.abstract",
        "citations.year",
        "references.title",
        "references.abstract",
        "references.year",
    ]

    # Full paper details
    COMPLETE = BASIC + AUTHOR_BASIC + CITATION_BASIC + ["url", "fieldsOfStudy", "publicationVenue", "publicationTypes"]


class CitationReferenceFields:
    """Common field combinations for citation and reference queries"""

    # Basic information
    BASIC = ["title"]

    # Citation/Reference context
    CONTEXT = ["contexts", "intents", "isInfluential"]

    # Paper details
    DETAILED = ["title", "abstract", "authors", "year", "venue", "url"]

    # Full information
    COMPLETE = CONTEXT + DETAILED

    # Valid fields for citation/reference queries
    VALID_FIELDS = {
        "contexts",
        "intents",
        "isInfluential",
        "title",
        "abstract",
        "authors",
        "year",
        "venue",
        "paperId",
        "url",
        "citationCount",
        "influentialCitationCount",
    }


# Configuration
class Config:
    # API Configuration
    API_VERSION = "v1"
    # Define rate limits (requests, seconds)
    SEARCH_LIMIT = (1, 1)  # 1 request per 1 second
    BATCH_LIMIT = (1, 1)  # 1 request per 1 second
    DEFAULT_LIMIT = (5, 1)  # 10 requests per 1 second

    # Endpoints categorization
    # These endpoints have stricter rate limits due to their computational intensity
    # and to prevent abuse of the recommendation system
    RESTRICTED_ENDPOINTS = [
        "/paper/batch",  # Batch operations are expensive
        "/paper/search",  # Search operations are computationally intensive
        "/recommendations",  # Recommendation generation is resource-intensive
    ]

    DYNAMIC_RATE_LIMITS: Dict[str, Tuple[int, int]] = {}

    BASE_URL = f"https://api.semanticscholar.org/graph/{API_VERSION}"
    TIMEOUT = 30  # seconds

    # Request Limits
    MAX_BATCH_SIZE = 100
    MAX_RESULTS_PER_PAGE = 100
    DEFAULT_PAGE_SIZE = 10
    MAX_BATCHES = 5

    # Fields Configuration
    DEFAULT_FIELDS = PaperFields.DEFAULT

    # Feature Flags
    ENABLE_CACHING = True
    DEBUG_MODE = False

    # Search Configuration
    SEARCH_TYPES = {
        "comprehensive": {
            "description": "Balanced search considering relevance and impact",
            "min_citations": None,
            "ranking_strategy": "balanced",
        },
        "influential": {
            "description": "Focus on highly-cited and influential papers",
            "min_citations": 50,
            "ranking_strategy": "citations",
        },
        "latest": {
            "description": "Focus on recent papers with impact",
            "min_citations": None,
            "ranking_strategy": "recency",
        },
    }


# Rate Limiter
class RateLimiter:
    def __init__(self):
        self._last_call_time = {}
        self._locks = {}
        # Use a dictionary to store per-endpoint rate limits
        self._endpoint_rate_limits = {}

    def _get_rate_limit(self, endpoint: str) -> Tuple[int, int]:
        # Check for dynamic rate limits first
        # if endpoint in RateLimitConfig.DYNAMIC_RATE_LIMITS:
        #     return RateLimitConfig.DYNAMIC_RATE_LIMITS[endpoint]

        # Use per-endpoint rate limits if available, otherwise fall back to defaults
        if endpoint in self._endpoint_rate_limits:
            return self._endpoint_rate_limits[endpoint]

        if any(restricted in endpoint for restricted in Config.RESTRICTED_ENDPOINTS):
            return Config.SEARCH_LIMIT
        return Config.DEFAULT_LIMIT

    # def _update_rate_limit(self, endpoint: str, headers: Dict):
    #     """Updates the dynamic rate limits based on response headers."""
    #     # Example header names (Semantic Scholar might use different ones)
    #     rate_limit_header = headers.get("X-RateLimit-Limit")
    #     rate_limit_remaining_header = headers.get("X-RateLimit-Remaining")
    #     rate_limit_reset_header = headers.get("X-RateLimit-Reset")  # Seconds until reset

    #     if rate_limit_header and rate_limit_reset_header:
    #         try:
    #             requests = int(rate_limit_header.split(",")[0])  # Assuming format like "5, 1"
    #             seconds = int(rate_limit_reset_header)
    #             RateLimitConfig.DYNAMIC_RATE_LIMITS[endpoint] = (requests, seconds)
    #             logger.info(f"Updated rate limit for {endpoint}: {requests} requests per {seconds} seconds")
    #         except ValueError:
    #             logger.warning(f"Could not parse rate limit headers for {endpoint}: {headers}")
    #     elif rate_limit_remaining_header is not None:
    #         # Could implement a more sophisticated approach based on remaining requests
    #         pass

    def _decrease_rate_limit(self, endpoint: str):
        """Decreases the rate limit for a specific endpoint (after a 429)."""
        current_rate_limit = self._get_rate_limit(endpoint)
        # Example: Double the delay (halve the requests per second)
        new_rate_limit = (current_rate_limit[0], current_rate_limit[1] * 2)
        self._endpoint_rate_limits[endpoint] = new_rate_limit
        logger.warning(f"Decreased rate limit for {endpoint} to: {new_rate_limit[0]} requests per {new_rate_limit[1]} seconds")

    async def acquire(self, endpoint: str):
        if endpoint not in self._locks:
            self._locks[endpoint] = asyncio.Lock()
            self._last_call_time[endpoint] = 0

        async with self._locks[endpoint]:
            rate_limit = self._get_rate_limit(endpoint)
            current_time = time.time()
            time_since_last_call = current_time - self._last_call_time[endpoint]
            logger.debug(f"time_since_last_call {time_since_last_call}")

            if time_since_last_call < rate_limit[1] / rate_limit[0]:  # Correct delay calculation
                delay = (rate_limit[1] / rate_limit[0]) - time_since_last_call
                logger.debug(f"Rate limiting: delaying {delay:.2f} seconds for {endpoint}")
                await asyncio.sleep(delay)

            self._last_call_time[endpoint] = time.time()


def create_error_response(
    error_type: ErrorType, message: str, details: Optional[Dict] = None, status_code: Optional[int] = None
) -> Dict:
    """Creates a standardized error response."""
    response = {
        "error": {
            "type": error_type.value,
            "message": message,
            "details": details or {},
        }
    }
    if status_code:
        response["error"]["status_code"] = status_code
    return response


rate_limiter = RateLimiter()


# Basic functions


def get_api_key() -> Optional[str]:
    """
    Get the Semantic Scholar API key from environment variables.

    Returns:
        The API key if found, otherwise None.  Logs warnings and instructions.
    """
    # api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    api_key = SEMANTIC_SCHOLAR_API_KEY

    if api_key:
        # Basic sanity check: Make sure the key isn't empty or whitespace
        if api_key.strip() == "":
            logger.warning("SEMANTIC_SCHOLAR_API_KEY is set but empty.")
            print("WARNING: SEMANTIC_SCHOLAR_API_KEY environment variable is empty.")  # User output
            return None  # Treat empty key as no key
        else:
            logger.info("Using Semantic Scholar API key from environment variable.")
            return api_key.strip()  # remove surronding whitespace
    else:
        logger.warning("No SEMANTIC_SCHOLAR_API_KEY set. Using unauthenticated access.")
        print("WARNING: No SEMANTIC_SCHOLAR_API_KEY set.  Using unauthenticated access (lower rate limits).")
        return None


async def handle_exception(loop, context):
    """Global exception handler for the event loop."""
    msg = context.get("exception", context["message"])
    logger.error(f"Caught exception: {msg}")
    # asyncio.create_task(shutdown()) # Removed shutdown as it's not relevant for a standalone script


async def initialize_client():
    """Initialize the global HTTP client."""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=Config.TIMEOUT, limits=httpx.Limits(max_keepalive_connections=10))
    return http_client


async def cleanup_client():
    """Cleanup the global HTTP client."""
    global http_client
    if http_client is not None:
        await http_client.aclose()
        http_client = None


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=60),
    retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.TimeoutException)),
)
@cached(ttl=600, cache=Cache.MEMORY, serializer=JsonSerializer(), key_builder=lambda f, e, p: f"{e}-{str(p)}")
async def make_request(endpoint: str, params: Dict = None) -> Dict:
    """Make a rate-limited request to the Semantic Scholar API."""
    try:
        # Apply rate limiting
        await rate_limiter.acquire(endpoint)

        # Get API key if available
        api_key = get_api_key()
        headers = {"x-api-key": api_key} if api_key else {}
        url = f"{Config.BASE_URL}{endpoint}"

        # Use global client
        client = await initialize_client()
        logger.debug(f"Making request to {url} with params: {params}")
        response = await client.get(url, params=params, headers=headers)
        response.raise_for_status()
        # rate_limiter._update_rate_limit(endpoint, response.headers)
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error {e.response.status_code} for {endpoint}: {e.response.text}")
        if e.response.status_code == 429:
            return create_error_response(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded. Consider using an API key for higher limits.",
                {
                    "retry_after": e.response.headers.get("retry-after"),
                    "authenticated": bool(get_api_key()),
                },
                rate_limiter._decrease_rate_limit(endpoint),
                status_code=e.response.status_code,
            )
        return create_error_response(
            ErrorType.API_ERROR,
            f"HTTP error: {e.response.status_code}",
            {"response": e.response.text},
            status_code=e.response.status_code,
        )
    except httpx.TimeoutException as e:
        logger.error(f"Request timeout for {endpoint}: {str(e)}")
        return create_error_response(ErrorType.TIMEOUT, f"Request timed out after {Config.TIMEOUT} seconds")
    except Exception as e:
        trace = traceback.format_exc()  # Get full traceback
        logger.exception(f"Unexpected error for {endpoint}: {str(e)}\n{trace}")
        return create_error_response(ErrorType.API_ERROR, str(e))


# 1. Paper Data Tools


# 1.1 Paper relevance search
async def paper_relevance_search(
    # context: Context, # Removed context
    query: str,
    fields: Optional[List[str]] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    year: Optional[str] = None,  # supports formats like "2019", "2016-2020", "2010-", "-2015"
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = Config.DEFAULT_PAGE_SIZE,
) -> Dict:
    """
    Search for papers on Semantic Scholar using relevance-based ranking.
    This endpoint is optimized for finding the most relevant papers matching a text query.
    Results are sorted by relevance score.

    Args:
        query (str): A text query to search for. The query will be matched against paper titles,
            abstracts, venue names, and author names. All terms in the query must be present
            in the paper for it to be returned. The query is case-insensitive and matches word
            prefixes (e.g. "quantum" matches "quantum" and "quantumly").

        fields (Optional[List[str]]): List of fields to return for each paper.
            paperId and title are always returned.
            Available fields:
            - abstract: The paper's abstract
            - authors: List of authors with name and authorId
            - citationCount: Total number of citations
            - citations: List of papers citing this paper
            - corpusId: Internal ID for the paper
            - embedding: Vector embedding of the paper
            - externalIds: External IDs (DOI, MAG, etc)
            - fieldsOfStudy: List of fields of study
            - influentialCitationCount: Number of influential citations
            - isOpenAccess: Whether paper is open access
            - openAccessPdf: Open access PDF URL if available
            - paperId: Semantic Scholar paper ID
            - publicationDate: Publication date in YYYY-MM-DD format
            - publicationTypes: List of publication types
            - publicationVenue: Venue information
            - references: List of papers cited by this paper
            - s2FieldsOfStudy: Semantic Scholar fields
            - title: Paper title
            - tldr: AI-generated TLDR summary
            - url: URL to Semantic Scholar paper page
            - venue: Publication venue name
            - year: Publication year

        publication_types (Optional[List[str]]): Filter by publication types.
            Available types:
            - Review
            - JournalArticle
            - CaseReport
            - ClinicalTrial
            - Conference
            - Dataset
            - Editorial
            - LettersAndComments
            - MetaAnalysis
            - News
            - Study
            - Book
            - BookSection

        open_access_pdf (bool): If True, only include papers with a public PDF.
            Default: False

        min_citation_count (Optional[int]): Minimum number of citations required.
            Papers with fewer citations will be filtered out.

        year (Optional[str]): Filter by publication year. Supports several formats:
            - Single year: "2019"
            - Year range: "2016-2020"
            - Since year: "2010-"
            - Until year: "-2015"

        venue (Optional[List[str]]): Filter by publication venues.
            Accepts full venue names or ISO4 abbreviations.
            Examples: ["Nature", "Science", "N. Engl. J. Med."]

        fields_of_study (Optional[List[str]]): Filter by fields of study.
            Available fields:
            - Computer Science
            - Medicine
            - Chemistry
            - Biology
            - Materials Science
            - Physics
            - Geology
            - Psychology
            - Art
            - History
            - Geography
            - Sociology
            - Business
            - Political Science
            - Economics
            - Philosophy
            - Mathematics
            - Engineering
            - Environmental Science
            - Agricultural and Food Sciences
            - Education
            - Law
            - Linguistics

        offset (int): Number of results to skip for pagination.
            Default: 0

        limit (int): Maximum number of results to return.
            Default: 10
            Maximum: 100

    Returns:
        Dict: {
            "total": int,      # Total number of papers matching the query
            "offset": int,     # Current offset in the results
            "next": int,       # Offset for the next page of results (if available)
            "data": List[Dict] # List of papers with requested fields
        }

    Notes:
        - Results are sorted by relevance to the query
        - All query terms must be present in the paper (AND operation)
        - Query matches are case-insensitive
        - Query matches word prefixes (e.g., "quantum" matches "quantum" and "quantumly")
        - Maximum of 100 results per request
        - Use offset parameter for pagination
        - Rate limits apply (see API documentation)
    """
    if not query.strip():
        return create_error_response(ErrorType.VALIDATION, "Query string cannot be empty")

    # Validate and prepare fields
    if fields is None:
        fields = PaperFields.DEFAULT
    else:
        invalid_fields = set(fields) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)},
            )

    # Validate and prepare parameters
    limit = min(limit, Config.MAX_RESULTS_PER_PAGE)
    params = {
        "query": query,
        "offset": offset,
        "limit": limit,
        "fields": ",".join(fields),
    }

    # Add optional filters
    if publication_types:
        params["publicationTypes"] = ",".join(publication_types)
    if open_access_pdf:
        params["openAccessPdf"] = "true"
    if min_citation_count is not None:
        params["minCitationCount"] = min_citation_count
    if year:
        params["year"] = year
    if venue:
        params["venue"] = ",".join(venue)
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    return await make_request("/paper/search", params)


# 1.2 Paper bulk search
async def paper_bulk_search(
    # context: Context, # Removed context
    query: Optional[str] = None,
    token: Optional[str] = None,
    fields: Optional[List[str]] = None,
    sort: Optional[str] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    publication_date_or_year: Optional[str] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
) -> Dict:
    """
    Bulk search for papers with advanced filtering and sorting options.
    Intended for retrieving large sets of papers efficiently.

    Args:
        query (Optional[str]): Text query to match against paper title and abstract.
            Supports boolean logic:
            - '+' for AND operation
            - '|' for OR operation
            - '-' to negate a term
            - '"' for phrase matching
            - '*' for prefix matching
            - '()' for precedence
            - '~N' for edit distance (default 2)
            Examples:
            - 'fish ladder' (contains both terms)
            - 'fish -ladder' (has fish, no ladder)
            - 'fish | ladder' (either term)
            - '"fish ladder"' (exact phrase)
            - '(fish ladder) | outflow'
            - 'fish~' (fuzzy match)
            - '"fish ladder"~3' (terms within 3 words)

        token (Optional[str]): Continuation token for pagination

        fields (Optional[List[str]]): Fields to return for each paper
            paperId is always returned
            Default: paperId and title only

        sort (Optional[str]): Sort order in format 'field:order'
            Fields: paperId, publicationDate, citationCount
            Order: asc (default), desc
            Default: 'paperId:asc'
            Examples:
            - 'publicationDate:asc' (oldest first)
            - 'citationCount:desc' (most cited first)

        publication_types (Optional[List[str]]): Filter by publication types:
            Review, JournalArticle, CaseReport, ClinicalTrial,
            Conference, Dataset, Editorial, LettersAndComments,
            MetaAnalysis, News, Study, Book, BookSection

        open_access_pdf (bool): Only include papers with public PDF

        min_citation_count (Optional[int]): Minimum citation threshold

        publication_date_or_year (Optional[str]): Date/year range filter
            Format: <startDate>:<endDate> in YYYY-MM-DD
            Supports partial dates and open ranges
            Examples:
            - '2019-03-05' (specific date)
            - '2019-03' (month)
            - '2019' (year)
            - '2016-03-05:2020-06-06' (range)
            - '1981-08-25:' (since date)
            - ':2015-01' (until date)

        year (Optional[str]): Publication year filter
            Examples: '2019', '2016-2020', '2010-', '-2015'

        venue (Optional[List[str]]): Filter by publication venues
            Accepts full names or ISO4 abbreviations
            Examples: ['Nature', 'N. Engl. J. Med.']

        fields_of_study (Optional[List[str]]): Filter by fields of study
            Available fields include: Computer Science, Medicine,
            Physics, Mathematics, etc.

    Returns:
        Dict: {
            'total': int,      # Total matching papers
            'token': str,      # Continuation token for next batch
            'data': List[Dict] # Papers with requested fields
        }

    Notes:
        - Returns up to 1,000 papers per call
        - Can fetch up to 10M papers total
        - Nested data (citations, references) not available
        - For larger datasets, use the Datasets API
    """
    # Build request parameters
    params = {}

    # Add query if provided
    if query:
        params["query"] = query.strip()

    # Add continuation token if provided
    if token:
        params["token"] = token

    # Add fields if provided
    if fields:
        # Validate fields
        invalid_fields = set(fields) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)},
            )
        params["fields"] = ",".join(fields)

    # Add sort if provided
    if sort:
        # Validate sort format
        valid_sort_fields = ["paperId", "publicationDate", "citationCount"]
        valid_sort_orders = ["asc", "desc"]

        try:
            field, order = sort.split(":")
            if field not in valid_sort_fields:
                return create_error_response(
                    ErrorType.VALIDATION,
                    f"Invalid sort field. Must be one of: {', '.join(valid_sort_fields)}",
                )
            if order not in valid_sort_orders:
                return create_error_response(
                    ErrorType.VALIDATION,
                    f"Invalid sort order. Must be one of: {', '.join(valid_sort_orders)}",
                )
            params["sort"] = sort
        except ValueError:
            return create_error_response(ErrorType.VALIDATION, "Sort must be in format 'field:order'")

    # Add publication types if provided
    if publication_types:
        valid_types = {
            "Review",
            "JournalArticle",
            "CaseReport",
            "ClinicalTrial",
            "Conference",
            "Dataset",
            "Editorial",
            "LettersAndComments",
            "MetaAnalysis",
            "News",
            "Study",
            "Book",
            "BookSection",
        }
        invalid_types = set(publication_types) - valid_types
        if invalid_types:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid publication types: {', '.join(invalid_types)}",
                {"valid_types": list(valid_types)},
            )
        params["publicationTypes"] = ",".join(publication_types)

    # Add open access PDF filter
    if open_access_pdf:
        params["openAccessPdf"] = "true"

    # Add minimum citation count if provided
    if min_citation_count is not None:
        if min_citation_count < 0:
            return create_error_response(ErrorType.VALIDATION, "Minimum citation count cannot be negative")
        params["minCitationCount"] = str(min_citation_count)

    # Add publication date/year if provided
    if publication_date_or_year:
        params["publicationDateOrYear"] = publication_date_or_year
    elif year:
        params["year"] = year

    # Add venue filter if provided
    if venue:
        params["venue"] = ",".join(venue)

    # Add fields of study filter if provided
    if fields_of_study:
        valid_fields = {
            "Computer Science",
            "Medicine",
            "Chemistry",
            "Biology",
            "Materials Science",
            "Physics",
            "Geology",
            "Psychology",
            "Art",
            "History",
            "Geography",
            "Sociology",
            "Business",
            "Political Science",
            "Economics",
            "Philosophy",
            "Mathematics",
            "Engineering",
            "Environmental Science",
            "Agricultural and Food Sciences",
            "Education",
            "Law",
            "Linguistics",
        }
        invalid_fields = set(fields_of_study) - valid_fields
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields of study: {', '.join(invalid_fields)}",
                {"valid_fields": list(valid_fields)},
            )
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    # Make the API request
    result = await make_request("/paper/search/bulk", params)

    # Handle potential errors
    if isinstance(result, Dict) and "error" in result:
        return result

    return result


# 1.3 Paper title search
async def paper_title_search(
    # context: Context, # Removed context
    query: str,
    fields: Optional[List[str]] = None,
    publication_types: Optional[List[str]] = None,
    open_access_pdf: bool = False,
    min_citation_count: Optional[int] = None,
    year: Optional[str] = None,
    venue: Optional[List[str]] = None,
    fields_of_study: Optional[List[str]] = None,
) -> Dict:
    """
    Find a single paper by title match. This endpoint is optimized for finding a specific paper
    by its title and returns the best matching paper based on title similarity.

    Args:
        query (str): The title text to search for. The query will be matched against paper titles
            to find the closest match. The match is case-insensitive and ignores punctuation.

        fields (Optional[List[str]]): List of fields to return for the paper.
            paperId and title are always returned.
            Available fields:
            - abstract: The paper's abstract
            - authors: List of authors with name and authorId
            - citationCount: Total number of citations
            - citations: List of papers citing this paper
            - corpusId: Internal ID for the paper
            - embedding: Vector embedding of the paper
            - externalIds: External IDs (DOI, MAG, etc)
            - fieldsOfStudy: List of fields of study
            - influentialCitationCount: Number of influential citations
            - isOpenAccess: Whether paper is open access
            - openAccessPdf: Open access PDF URL if available
            - paperId: Semantic Scholar paper ID
            - publicationDate: Publication date in YYYY-MM-DD format
            - publicationTypes: List of publication types
            - publicationVenue: Venue information
            - references: List of papers cited by this paper
            - s2FieldsOfStudy: Semantic Scholar fields
            - title: Paper title
            - tldr: AI-generated TLDR summary
            - url: URL to Semantic Scholar paper page
            - venue: Publication venue name
            - year: Publication year

        publication_types (Optional[List[str]]): Filter by publication types.
            Available types:
            - Review
            - JournalArticle
            - CaseReport
            - ClinicalTrial
            - Conference
            - Dataset
            - Editorial
            - LettersAndComments
            - MetaAnalysis
            - News
            - Study
            - Book
            - BookSection

        open_access_pdf (bool): If True, only include papers with a public PDF.
            Default: False

        min_citation_count (Optional[int]): Minimum number of citations required.
            Papers with fewer citations will be filtered out.

        year (Optional[str]): Filter by publication year. Supports several formats:
            - Single year: "2019"
            - Year range: "2016-2020"
            - Since year: "2010-"
            - Until year: "-2015"

        venue (Optional[List[str]]): Filter by publication venues.
            Accepts full venue names or ISO4 abbreviations.
            Examples: ["Nature", "Science", "N. Engl. J. Med."]

        fields_of_study (Optional[List[str]]): Filter by fields of study.
            Available fields:
            - Computer Science
            - Medicine
            - Chemistry
            - Biology
            - Materials Science
            - Physics
            - Geology
            - Psychology
            - Art
            - History
            - Geography
            - Sociology
            - Business
            - Political Science
            - Economics
            - Philosophy
            - Mathematics
            - Engineering
            - Environmental Science
            - Agricultural and Food Sciences
            - Education
            - Law
            - Linguistics

    Returns:
        Dict: {
            "paperId": str,      # Semantic Scholar Paper ID
            "title": str,        # Paper title
            "matchScore": float, # Similarity score between query and matched title
            ...                  # Additional requested fields
        }

        Returns error response if no matching paper is found.

    Notes:
        - Returns the single best matching paper based on title similarity
        - Match score indicates how well the title matches the query
        - Case-insensitive matching
        - Ignores punctuation in matching
        - Filters are applied after finding the best title match
    """
    if not query.strip():
        return create_error_response(ErrorType.VALIDATION, "Query string cannot be empty")

    # Validate and prepare fields
    if fields is None:
        fields = PaperFields.DEFAULT
    else:
        invalid_fields = set(fields) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)},
            )

    # Build base parameters
    params = {"query": query}

    # Add optional parameters
    if fields:
        params["fields"] = ",".join(fields)
    if publication_types:
        params["publicationTypes"] = ",".join(publication_types)
    if open_access_pdf:
        params["openAccessPdf"] = "true"
    if min_citation_count is not None:
        params["minCitationCount"] = str(min_citation_count)
    if year:
        params["year"] = year
    if venue:
        params["venue"] = ",".join(venue)
    if fields_of_study:
        params["fieldsOfStudy"] = ",".join(fields_of_study)

    result = await make_request("/paper/search/match", params)

    # Handle specific error cases
    if isinstance(result, Dict):
        if "error" in result:
            error_msg = result["error"].get("message", "")
            if "404" in error_msg:
                return create_error_response(
                    ErrorType.VALIDATION,
                    "No matching paper found",
                    {"original_query": query},
                )
            return result

    return result


# 1.4 Details about a paper
async def paper_details(
    # context: Context, # Removed context
    paper_id: str,
    fields: Optional[List[str]] = None,
) -> Dict:
    """
    Get details about a paper using various types of identifiers.
    This endpoint provides comprehensive metadata about a paper.

    Args:
        paper_id (str): Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")
              Supported URLs from: semanticscholar.org, arxiv.org, aclweb.org,
                                 acm.org, biorxiv.org

        fields (Optional[List[str]]): List of fields to return.
            paperId is always returned.
            Available fields:
            - abstract: The paper's abstract
            - authors: List of authors with name and authorId
            - citationCount: Total number of citations
            - citations: List of papers citing this paper
            - corpusId: Internal ID for the paper
            - embedding: Vector embedding of the paper
            - externalIds: External IDs (DOI, MAG, etc)
            - fieldsOfStudy: List of fields of study
            - influentialCitationCount: Number of influential citations
            - isOpenAccess: Whether paper is open access
            - openAccessPdf: Open access PDF URL if available
            - paperId: Semantic Scholar paper ID
            - publicationDate: Publication date in YYYY-MM-DD format
            - publicationTypes: List of publication types
            - publicationVenue: Venue information
            - references: List of papers cited by this paper
            - s2FieldsOfStudy: Semantic Scholar fields
            - title: Paper title
            - tldr: AI-generated TLDR summary
            - url: URL to Semantic Scholar paper page
            - venue: Publication venue name
            - year: Publication year

            Special syntax for nested fields:
            - For citations/references: citations.title, references.abstract, etc.
            - For authors: authors.name, authors.affiliations, etc.
            - For embeddings: embedding.specter_v2 for v2 embeddings

            If omitted, returns only paperId and title.

    Returns:
        Dict: Paper details with requested fields.
            Always includes paperId.
            Returns error response if paper not found.

    Notes:
        - Supports multiple identifier types for flexibility
        - Nested fields available for detailed citation/reference/author data
        - Rate limits apply (see API documentation)
        - Some fields may be null if data is not available
    """
    if not paper_id.strip():
        return create_error_response(ErrorType.VALIDATION, "Paper ID cannot be empty")

    # Build request parameters
    params = {}
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/paper/{paper_id}", params)

    # Handle potential errors
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(ErrorType.VALIDATION, "Paper not found", {"paper_id": paper_id})
        return result

    return result


# 1.5 Get details for multiple papers at once
async def paper_batch_details(
    # context: Context, # Removed context
    paper_ids: List[str],
    fields: Optional[str] = None,
) -> Dict:
    """
    Get details for multiple papers in a single batch request.
    This endpoint is optimized for efficiently retrieving details about known papers.

    Args:
        paper_ids (List[str]): List of paper identifiers. Each ID can be in any of these formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")
              Supported URLs from: semanticscholar.org, arxiv.org, aclweb.org,
                                 acm.org, biorxiv.org
            Maximum: 500 IDs per request

        fields (Optional[str]): Comma-separated list of fields to return for each paper.
            paperId is always returned.
            Available fields:
            - abstract: The paper's abstract
            - authors: List of authors with name and authorId
            - citationCount: Total number of citations
            - citations: List of papers citing this paper
            - corpusId: Internal ID for the paper
            - embedding: Vector embedding of the paper
            - externalIds: External IDs (DOI, MAG, etc)
            - fieldsOfStudy: List of fields of study
            - influentialCitationCount: Number of influential citations
            - isOpenAccess: Whether paper is open access
            - openAccessPdf: Open access PDF URL if available
            - paperId: Semantic Scholar paper ID
            - publicationDate: Publication date in YYYY-MM-DD format
            - publicationTypes: List of publication types
            - publicationVenue: Venue information
            - references: List of papers cited by this paper
            - s2FieldsOfStudy: Semantic Scholar fields
            - title: Paper title
            - tldr: AI-generated TLDR summary
            - url: URL to Semantic Scholar paper page
            - venue: Publication venue name
            - year: Publication year

            Special syntax for nested fields:
            - For citations/references: citations.title, references.abstract, etc.
            - For authors: authors.name, authors.affiliations, etc.
            - For embeddings: embedding.specter_v2 for v2 embeddings

            If omitted, returns only paperId and title.

    Returns:
        List[Dict]: List of paper details with requested fields.
            - Results maintain the same order as input paper_ids
            - Invalid or not found paper IDs return null in the results
            - Each paper object contains the requested fields
            - paperId is always included in each paper object

    Notes:
        - More efficient than making multiple single-paper requests
        - Maximum of 500 paper IDs per request
        - Rate limits apply (see API documentation)
        - Some fields may be null if data is not available
        - Invalid paper IDs return null instead of causing an error
        - Order of results matches order of input IDs for easy mapping
    """
    # Validate inputs
    if not paper_ids:
        return create_error_response(ErrorType.VALIDATION, "Paper IDs list cannot be empty")

    if len(paper_ids) > 500:
        return create_error_response(
            ErrorType.VALIDATION,
            "Cannot process more than 500 paper IDs at once",
            {"max_papers": 500, "received": len(paper_ids)},
        )

    # Validate fields if provided
    if fields:
        field_list = fields.split(",")
        invalid_fields = set(field_list) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)},
            )

    # Build request parameters
    params = {}
    if fields:
        params["fields"] = fields

    # Make POST request with proper structure
    try:
        async with httpx.AsyncClient(timeout=Config.TIMEOUT) as client:
            api_key = get_api_key()
            headers = {"x-api-key": api_key} if api_key else {}

            response = await client.post(
                f"{Config.BASE_URL}/paper/batch",
                params=params,
                json={"ids": paper_ids},
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return create_error_response(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded",
                {"retry_after": e.response.headers.get("retry-after")},
            )
        return create_error_response(
            ErrorType.API_ERROR,
            f"HTTP error: {e.response.status_code}",
            {"response": e.response.text},
        )
    except httpx.TimeoutException:
        return create_error_response(ErrorType.TIMEOUT, f"Request timed out after {Config.TIMEOUT} seconds")
    except Exception as e:
        return create_error_response(ErrorType.API_ERROR, str(e))


# 1.6 Details about a paper's authors
async def paper_authors(
    # context: Context, # Removed context
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """
    Get details about the authors of a paper with pagination support.
    This endpoint provides author information and their contributions.

    Args:
        paper_id (str): Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")

        fields (Optional[List[str]]): List of fields to return for each author.
            authorId is always returned.
            Available fields:
            - name: Author's name
            - aliases: Alternative names for the author
            - affiliations: List of author's affiliations
            - homepage: Author's homepage URL
            - paperCount: Total number of papers by this author
            - citationCount: Total citations received by this author
            - hIndex: Author's h-index
            - papers: List of papers by this author (returns paperId and title)

            Special syntax for paper fields:
            - papers.year: Include year for each paper
            - papers.authors: Include authors for each paper
            - papers.abstract: Include abstract for each paper
            - papers.venue: Include venue for each paper
            - papers.citations: Include citation count for each paper

            If omitted, returns only authorId and name.

        offset (int): Number of authors to skip for pagination.
            Default: 0

        limit (int): Maximum number of authors to return.
            Default: 100
            Maximum: 1000

    Returns:
        Dict: {
            "offset": int,     # Current offset in the results
            "next": int,       # Next offset (if more results available)
            "data": List[Dict] # List of authors with requested fields
        }

    Notes:
        - Authors are returned in the order they appear on the paper
        - Supports pagination for papers with many authors
        - Some fields may be null if data is not available
        - Rate limits apply (see API documentation)
    """
    if not paper_id.strip():
        return create_error_response(ErrorType.VALIDATION, "Paper ID cannot be empty")

    # Validate limit
    if limit > 1000:
        return create_error_response(ErrorType.VALIDATION, "Limit cannot exceed 1000", {"max_limit": 1000})

    # Validate fields
    if fields:
        invalid_fields = set(fields) - AuthorDetailFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(AuthorDetailFields.VALID_FIELDS)},
            )

    # Build request parameters
    params = {"offset": offset, "limit": limit}
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/paper/{paper_id}/authors", params)

    # Handle potential errors
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(ErrorType.VALIDATION, "Paper not found", {"paper_id": paper_id})
        return result

    return result


# 1.7 Details about a paper's citations
async def paper_citations(
    # context: Context, # Removed context
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """
    Get papers that cite the specified paper (papers where this paper appears in their bibliography).
    This endpoint provides detailed citation information including citation contexts.

    Args:
        paper_id (str): Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")

        fields (Optional[List[str]]): List of fields to return for each citing paper.
            paperId is always returned.
            Available fields:
            - title: Paper title
            - abstract: Paper abstract
            - year: Publication year
            - venue: Publication venue
            - authors: List of authors
            - url: URL to paper page
            - citationCount: Number of citations received
            - influentialCitationCount: Number of influential citations

            Citation-specific fields:
            - contexts: List of citation contexts (text snippets)
            - intents: List of citation intents (Background, Method, etc.)
            - isInfluential: Whether this is an influential citation

            If omitted, returns only paperId and title.

        offset (int): Number of citations to skip for pagination.
            Default: 0

        limit (int): Maximum number of citations to return.
            Default: 100
            Maximum: 1000

    Returns:
        Dict: {
            "offset": int,     # Current offset in the results
            "next": int,       # Next offset (if more results available)
            "data": List[Dict] # List of citing papers with requested fields
        }

    Notes:
        - Citations are sorted by citation date (newest first)
        - Includes citation context when available
        - Supports pagination for highly-cited papers
        - Some fields may be null if data is not available
        - Rate limits apply (see API documentation)
    """
    if not paper_id.strip():
        return create_error_response(ErrorType.VALIDATION, "Paper ID cannot be empty")

    # Validate limit
    if limit > 1000:
        return create_error_response(ErrorType.VALIDATION, "Limit cannot exceed 1000", {"max_limit": 1000})

    # Validate fields
    if fields:
        invalid_fields = set(fields) - CitationReferenceFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(CitationReferenceFields.VALID_FIELDS)},
            )

    # Build request parameters
    params = {"offset": offset, "limit": limit}
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/paper/{paper_id}/citations", params)

    # Handle potential errors
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(ErrorType.VALIDATION, "Paper not found", {"paper_id": paper_id})
        return result

    return result


# 1.8 Details about a paper's references
async def paper_references(
    # context: Context, # Removed context
    paper_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """
    Get papers cited by the specified paper (papers appearing in this paper's bibliography).
    This endpoint provides detailed reference information including citation contexts.

    Args:
        paper_id (str): Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")

        fields (Optional[List[str]]): List of fields to return for each referenced paper.
            paperId is always returned.
            Available fields:
            - title: Paper title
            - abstract: Paper abstract
            - year: Publication year
            - venue: Publication venue
            - authors: List of authors
            - url: URL to paper page
            - citationCount: Number of citations received
            - influentialCitationCount: Number of influential citations

            Reference-specific fields:
            - contexts: List of citation contexts (text snippets)
            - intents: List of citation intents (Background, Method, etc.)
            - isInfluential: Whether this is an influential citation

            If omitted, returns only paperId and title.

        offset (int): Number of references to skip for pagination.
            Default: 0

        limit (int): Maximum number of references to return.
            Default: 100
            Maximum: 1000

    Returns:
        Dict: {
            "offset": int,     # Current offset in the results
            "next": int,       # Next offset (if more results available)
            "data": List[Dict] # List of referenced papers with requested fields
        }

    Notes:
        - References are returned in the order they appear in the bibliography
        - Includes citation context when available
        - Supports pagination for papers with many references
        - Some fields may be null if data is not available
        - Rate limits apply (see API documentation)
    """
    if not paper_id.strip():
        return create_error_response(ErrorType.VALIDATION, "Paper ID cannot be empty")

    # Validate limit
    if limit > 1000:
        return create_error_response(ErrorType.VALIDATION, "Limit cannot exceed 1000", {"max_limit": 1000})

    # Validate fields
    if fields:
        invalid_fields = set(fields) - CitationReferenceFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(CitationReferenceFields.VALID_FIELDS)},
            )

    # Build request parameters
    params = {"offset": offset, "limit": limit}
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/paper/{paper_id}/references", params)

    # Handle potential errors
    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(ErrorType.VALIDATION, "Paper not found", {"paper_id": paper_id})
        return result

    return result


# 2. Author Data Tools


# 2.1 Search for authors by name
async def author_search(
    # context: Context, # Removed context
    query: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """
    Search for authors by name on Semantic Scholar.
    This endpoint is optimized for finding authors based on their name.
    Results are sorted by relevance to the query.

    Args:
        query (str): The name text to search for. The query will be matched against author names
            and their known aliases. The match is case-insensitive and matches name prefixes.
            Examples:
            - "Albert Einstein"
            - "Einstein, Albert"
            - "A Einstein"

        fields (Optional[List[str]]): List of fields to return for each author.
            authorId is always returned.
            Available fields:
            - name: Author's name
            - aliases: Alternative names for the author
            - url: URL to author's S2 profile
            - affiliations: List of author's affiliations
            - homepage: Author's homepage URL
            - paperCount: Total number of papers by this author
            - citationCount: Total citations received by this author
            - hIndex: Author's h-index
            - papers: List of papers by this author (returns paperId and title)

            Special syntax for paper fields:
            - papers.year: Include year for each paper
            - papers.authors: Include authors for each paper
            - papers.abstract: Include abstract for each paper
            - papers.venue: Include venue for each paper
            - papers.citations: Include citation count for each paper

            If omitted, returns only authorId and name.

        offset (int): Number of authors to skip for pagination.
            Default: 0

        limit (int): Maximum number of authors to return.
            Default: 100
            Maximum: 1000

    Returns:
        Dict: {
            "total": int,      # Total number of authors matching the query
            "offset": int,     # Current offset in the results
            "next": int,       # Next offset (if more results available)
            "data": List[Dict] # List of authors with requested fields
        }

    Notes:
        - Results are sorted by relevance to the query
        - Matches against author names and aliases
        - Case-insensitive matching
        - Matches name prefixes
        - Supports pagination for large result sets
        - Some fields may be null if data is not available
        - Rate limits apply (see API documentation)
    """
    if not query.strip():
        return create_error_response(ErrorType.VALIDATION, "Query string cannot be empty")

    # Validate limit
    if limit > 1000:
        return create_error_response(ErrorType.VALIDATION, "Limit cannot exceed 1000", {"max_limit": 1000})

    # Validate fields
    if fields:
        invalid_fields = set(fields) - AuthorDetailFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(AuthorDetailFields.VALID_FIELDS)},
            )

    # Build request parameters
    params = {"query": query, "offset": offset, "limit": limit}
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    return await make_request("/author/search", params)


# 2.2 Details about an author
async def author_details(
    # context: Context, # Removed context
    author_id: str,
    fields: Optional[List[str]] = None,
) -> Dict:
    """
    Get detailed information about an author by their ID.
    This endpoint provides comprehensive metadata about an author.

    Args:
        author_id (str): Semantic Scholar author ID.
            This is a unique identifier assigned by Semantic Scholar.
            Example: "1741101" (Albert Einstein)

        fields (Optional[List[str]]): List of fields to return.
            authorId is always returned.
            Available fields:
            - name: Author's name
            - aliases: Alternative names for the author
            - url: URL to author's S2 profile
            - affiliations: List of author's affiliations
            - homepage: Author's homepage URL
            - paperCount: Total number of papers by this author
            - citationCount: Total citations received by this author
            - hIndex: Author's h-index
            - papers: List of papers by this author (returns paperId and title)

            Special syntax for paper fields:
            - papers.year: Include year for each paper
            - papers.authors: Include authors for each paper
            - papers.abstract: Include abstract for each paper
            - papers.venue: Include venue for each paper
            - papers.citations: Include citation count for each paper

            If omitted, returns only authorId and name.

    Returns:
        Dict: Author details with requested fields.
            Always includes authorId.
            Returns error response if author not found.

    Notes:
        - Provides comprehensive author metadata
        - Papers list is limited to most recent papers
        - For complete paper list, use author_papers endpoint
        - Some fields may be null if data is not available
        - Rate limits apply (see API documentation)
    """
    if not author_id.strip():
        return create_error_response(ErrorType.VALIDATION, "Author ID cannot be empty")

    # Validate fields
    if fields:
        invalid_fields = set(fields) - AuthorDetailFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(AuthorDetailFields.VALID_FIELDS)},
            )

    # Build request parameters
    params = {}
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/author/{author_id}", params)

    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(ErrorType.VALIDATION, "Author not found", {"author_id": author_id})
        return result

    return result


# 2.3 Details about an author's papers
async def author_papers(
    # context: Context, # Removed context
    author_id: str,
    fields: Optional[List[str]] = None,
    offset: int = 0,
    limit: int = 100,
) -> Dict:
    """
    Get papers written by an author with pagination support.
    This endpoint provides detailed information about an author's publications.

    Args:
        author_id (str): Semantic Scholar author ID.
            This is a unique identifier assigned by Semantic Scholar.
            Example: "1741101" (Albert Einstein)

        fields (Optional[List[str]]): List of fields to return for each paper.
            paperId is always returned.
            Available fields:
            - title: Paper title
            - abstract: Paper abstract
            - year: Publication year
            - venue: Publication venue
            - authors: List of authors
            - url: URL to paper page
            - citationCount: Number of citations received
            - influentialCitationCount: Number of influential citations
            - isOpenAccess: Whether paper is open access
            - openAccessPdf: Open access PDF URL if available
            - fieldsOfStudy: List of fields of study
            - s2FieldsOfStudy: Semantic Scholar fields
            - publicationTypes: List of publication types
            - publicationDate: Publication date in YYYY-MM-DD format
            - journal: Journal information
            - externalIds: External IDs (DOI, MAG, etc)

            If omitted, returns only paperId and title.

        offset (int): Number of papers to skip for pagination.
            Default: 0

        limit (int): Maximum number of papers to return.
            Default: 100
            Maximum: 1000

    Returns:
        Dict: {
            "offset": int,     # Current offset in the results
            "next": int,       # Next offset (if more results available)
            "data": List[Dict] # List of papers with requested fields
        }

    Notes:
        - Papers are sorted by publication date (newest first)
        - Supports pagination for authors with many papers
        - Some fields may be null if data is not available
        - Rate limits apply (see API documentation)
    """
    if not author_id.strip():
        return create_error_response(ErrorType.VALIDATION, "Author ID cannot be empty")

    # Validate limit
    if limit > 1000:
        return create_error_response(ErrorType.VALIDATION, "Limit cannot exceed 1000", {"max_limit": 1000})

    # Validate fields
    if fields:
        invalid_fields = set(fields) - PaperFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(PaperFields.VALID_FIELDS)},
            )

    # Build request parameters
    params = {"offset": offset, "limit": limit}
    if fields:
        params["fields"] = ",".join(fields)

    # Make the API request
    result = await make_request(f"/author/{author_id}/papers", params)

    if isinstance(result, Dict) and "error" in result:
        error_msg = result["error"].get("message", "")
        if "404" in error_msg:
            return create_error_response(ErrorType.VALIDATION, "Author not found", {"author_id": author_id})
        return result

    return result


# 2.4 Get details for multiple authors at once
async def author_batch_details(
    # context: Context, # Removed context
    author_ids: List[str],
    fields: Optional[str] = None,
) -> Dict:
    """
    Get details for multiple authors in a single batch request.
    This endpoint is optimized for efficiently retrieving details about known authors.

    Args:
        author_ids (List[str]): List of Semantic Scholar author IDs.
            These are unique identifiers assigned by Semantic Scholar.
            Example: ["1741101", "1741102"]
            Maximum: 1000 IDs per request

        fields (Optional[str]): Comma-separated list of fields to return for each author.
            authorId is always returned.
            Available fields:
            - name: Author's name
            - aliases: Alternative names for the author
                        - url: URL to author's S2 profile
            - affiliations: List of author's affiliations
            - homepage: Author's homepage URL
            - paperCount: Total number of papers by this author
            - citationCount: Total citations received by this author
            - hIndex: Author's h-index
            - papers: List of papers by this author (returns paperId and title)

            Special syntax for paper fields:
            - papers.year: Include year for each paper
            - papers.authors: Include authors for each paper
            - papers.abstract: Include abstract for each paper
            - papers.venue: Include venue for each paper
            - papers.citations: Include citation count for each paper

            If omitted, returns only authorId and name.

    Returns:
        List[Dict]: List of author details with requested fields.
            - Results maintain the same order as input author_ids
            - Invalid or not found author IDs return null in the results
            - Each author object contains the requested fields
            - authorId is always included in each author object

    Notes:
        - More efficient than making multiple single-author requests
        - Maximum of 1000 author IDs per request
        - Rate limits apply (see API documentation)
        - Some fields may be null if data is not available
        - Invalid author IDs return null instead of causing an error
        - Order of results matches order of input IDs for easy mapping
    """
    # Validate inputs
    if not author_ids:
        return create_error_response(ErrorType.VALIDATION, "Author IDs list cannot be empty")

    if len(author_ids) > 1000:
        return create_error_response(
            ErrorType.VALIDATION,
            "Cannot process more than 1000 author IDs at once",
            {"max_authors": 1000, "received": len(author_ids)},
        )

    # Validate fields if provided
    if fields:
        field_list = fields.split(",")
        invalid_fields = set(field_list) - AuthorDetailFields.VALID_FIELDS
        if invalid_fields:
            return create_error_response(
                ErrorType.VALIDATION,
                f"Invalid fields: {', '.join(invalid_fields)}",
                {"valid_fields": list(AuthorDetailFields.VALID_FIELDS)},
            )

    # Build request parameters
    params = {}
    if fields:
        params["fields"] = fields

    # Make POST request with proper structure
    try:
        async with httpx.AsyncClient(timeout=Config.TIMEOUT) as client:
            api_key = get_api_key()
            headers = {"x-api-key": api_key} if api_key else {}

            response = await client.post(
                f"{Config.BASE_URL}/author/batch",
                params=params,
                json={"ids": author_ids},
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return create_error_response(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded",
                {"retry_after": e.response.headers.get("retry-after")},
            )
        return create_error_response(
            ErrorType.API_ERROR,
            f"HTTP error: {e.response.status_code}",
            {"response": e.response.text},
        )
    except httpx.TimeoutException:
        return create_error_response(ErrorType.TIMEOUT, f"Request timed out after {Config.TIMEOUT} seconds")
    except Exception as e:
        return create_error_response(ErrorType.API_ERROR, str(e))


# 3. Paper Recommendation Tools


# 3.1 Get recommendations based on a single paper
async def get_paper_recommendations_single(
    # context: Context, # Removed context
    paper_id: str,
    fields: Optional[str] = None,
    limit: int = 100,
    from_pool: str = "recent",
) -> Dict:
    """
    Get paper recommendations based on a single seed paper.
    This endpoint is optimized for finding papers similar to a specific paper.

    Args:
        paper_id (str): Paper identifier in one of the following formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")

        fields (Optional[str]): Comma-separated list of fields to return for each paper.
            paperId is always returned.
            Available fields:
            - title: Paper title
            - abstract: Paper abstract
            - year: Publication year
            - venue: Publication venue
            - authors: List of authors
            - url: URL to paper page
            - citationCount: Number of citations received
            - influentialCitationCount: Number of influential citations
            - isOpenAccess: Whether paper is open access
            - openAccessPdf: Open access PDF URL if available
            - fieldsOfStudy: List of fields of study
            - publicationTypes: List of publication types
            - publicationDate: Publication date in YYYY-MM-DD format
            - journal: Journal information
            - externalIds: External IDs (DOI, MAG, etc)

            If omitted, returns only paperId and title.

        limit (int): Maximum number of recommendations to return.
            Default: 100
            Maximum: 500

        from_pool (str): Which pool of papers to recommend from.
            Options:
            - "recent": Recent papers (default)
            - "all-cs": All computer science papers
            Default: "recent"

    Returns:
        Dict: {
            "recommendedPapers": List[Dict] # List of recommended papers with requested fields
        }

    Notes:
        - Recommendations are based on content similarity and citation patterns
        - Results are sorted by relevance to the seed paper
        - "recent" pool focuses on papers from the last few years
        - "all-cs" pool includes older computer science papers
        - Rate limits apply (see API documentation)
        - Some fields may be null if data is not available
    """
    try:
        # Apply rate limiting
        endpoint = "/recommendations"
        await rate_limiter.acquire(endpoint)

        # Validate limit
        if limit > 500:
            return create_error_response(
                ErrorType.VALIDATION,
                "Cannot request more than 500 recommendations",
                {"max_limit": 500, "requested": limit},
            )

        # Validate pool
        if from_pool not in ["recent", "all-cs"]:
            return create_error_response(
                ErrorType.VALIDATION,
                "Invalid paper pool specified",
                {"valid_pools": ["recent", "all-cs"]},
            )

        # Build request parameters
        params = {"limit": limit, "from": from_pool}
        if fields:
            params["fields"] = fields

        # Make the API request
        async with httpx.AsyncClient(timeout=Config.TIMEOUT) as client:
            api_key = get_api_key()
            headers = {"x-api-key": api_key} if api_key else {}

            url = f"https://api.semanticscholar.org/recommendations/v1/papers/forpaper/{paper_id}"
            response = await client.get(url, params=params, headers=headers)

            # Handle specific error cases
            if response.status_code == 404:
                return create_error_response(ErrorType.VALIDATION, "Paper not found", {"paper_id": paper_id})

            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return create_error_response(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded. Consider using an API key for higher limits.",
                {
                    "retry_after": e.response.headers.get("retry-after"),
                    "authenticated": bool(get_api_key()),
                },
            )
        return create_error_response(
            ErrorType.API_ERROR,
            f"HTTP error {e.response.status_code}",
            {"response": e.response.text},
        )
    except httpx.TimeoutException:
        return create_error_response(ErrorType.TIMEOUT, f"Request timed out after {Config.TIMEOUT} seconds")
    except Exception as e:
        logger.error(f"Unexpected error in recommendations: {str(e)}")
        return create_error_response(ErrorType.API_ERROR, "Failed to get recommendations", {"error": str(e)})


# 3.2 Get recommendations based on multiple papers
async def get_paper_recommendations_multi(
    # context: Context, # Removed context
    positive_paper_ids: List[str],
    negative_paper_ids: Optional[List[str]] = None,
    fields: Optional[str] = None,
    limit: int = 100,
) -> Dict:
    """
    Get paper recommendations based on multiple positive and optional negative examples.
    This endpoint is optimized for finding papers similar to a set of papers while
    avoiding papers similar to the negative examples.

    Args:
        positive_paper_ids (List[str]): List of paper IDs to use as positive examples.
            Papers similar to these will be recommended.
            Each ID can be in any of these formats:
            - Semantic Scholar ID (e.g., "649def34f8be52c8b66281af98ae884c09aef38b")
            - CorpusId:<id> (e.g., "CorpusId:215416146")
            - DOI:<doi> (e.g., "DOI:10.18653/v1/N18-3011")
            - ARXIV:<id> (e.g., "ARXIV:2106.15928")
            - MAG:<id> (e.g., "MAG:112218234")
            - ACL:<id> (e.g., "ACL:W12-3903")
            - PMID:<id> (e.g., "PMID:19872477")
            - PMCID:<id> (e.g., "PMCID:2323736")
            - URL:<url> (e.g., "URL:https://arxiv.org/abs/2106.15928v1")

        negative_paper_ids (Optional[List[str]]): List of paper IDs to use as negative examples.
            Papers similar to these will be avoided in recommendations.
            Uses same ID formats as positive_paper_ids.

        fields (Optional[str]): Comma-separated list of fields to return for each paper.
            paperId is always returned.
            Available fields:
            - title: Paper title
            - abstract: Paper abstract
            - year: Publication year
            - venue: Publication venue
            - authors: List of authors
            - url: URL to paper page
            - citationCount: Number of citations received
            - influentialCitationCount: Number of influential citations
            - isOpenAccess: Whether paper is open access
            - openAccessPdf: Open access PDF URL if available
            - fieldsOfStudy: List of fields of study
            - publicationTypes: List of publication types
            - publicationDate: Publication date in YYYY-MM-DD format
            - journal: Journal information
            - externalIds: External IDs (DOI, MAG, etc)

            If omitted, returns only paperId and title.

        limit (int): Maximum number of recommendations to return.
            Default: 100
            Maximum: 500

    Returns:
        Dict: {
            "recommendedPapers": List[Dict] # List of recommended papers with requested fields
        }

    Notes:
        - Recommendations balance similarity to positive examples and dissimilarity to negative examples
        - Results are sorted by relevance score
        - More positive examples can help focus recommendations
        - Negative examples help filter out unwanted topics/approaches
        - Rate limits apply (see API documentation)
        - Some fields may be null if data is not available
    """
    try:
        # Apply rate limiting
        endpoint = "/recommendations"
        await rate_limiter.acquire(endpoint)

        # Validate inputs
        if not positive_paper_ids:
            return create_error_response(ErrorType.VALIDATION, "Must provide at least one positive paper ID")

        if limit > 500:
            return create_error_response(
                ErrorType.VALIDATION,
                "Cannot request more than 500 recommendations",
                {"max_limit": 500, "requested": limit},
            )

        # Build request parameters
        params = {"limit": limit}
        if fields:
            params["fields"] = fields

        request_body = {
            "positivePaperIds": positive_paper_ids,
            "negativePaperIds": negative_paper_ids or [],
        }

        # Make the API request
        async with httpx.AsyncClient(timeout=Config.TIMEOUT) as client:
            api_key = get_api_key()
            headers = {"x-api-key": api_key} if api_key else {}

            url = "https://api.semanticscholar.org/recommendations/v1/papers"
            response = await client.post(url, params=params, json=request_body, headers=headers)

            # Handle specific error cases
            if response.status_code == 404:
                return create_error_response(
                    ErrorType.VALIDATION,
                    "One or more input papers not found",
                    {
                        "positive_ids": positive_paper_ids,
                        "negative_ids": negative_paper_ids,
                    },
                )

            response.raise_for_status()
            return response.json()

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            return create_error_response(
                ErrorType.RATE_LIMIT,
                "Rate limit exceeded. Consider using an API key for higher limits.",
                {
                    "retry_after": e.response.headers.get("retry-after"),
                    "authenticated": bool(get_api_key()),
                },
            )
        return create_error_response(
            ErrorType.API_ERROR,
            f"HTTP error {e.response.status_code}",
            {"response": e.response.text},
        )
    except httpx.TimeoutException:
        return create_error_response(ErrorType.TIMEOUT, f"Request timed out after {Config.TIMEOUT} seconds")
    except Exception as e:
        logger.error(f"Unexpected error in recommendations: {str(e)}")
        return create_error_response(ErrorType.API_ERROR, "Failed to get recommendations", {"error": str(e)})


async def scrape_pdf_link(doi: str) -> Optional[str]:
    """
    Extracts a direct PDF link by scraping the final article webpage.

    Args:
        paper_url: The initial article URL (could be a DOI link).

    Returns:
        The direct PDF URL if found, otherwise None.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
        "Referer": "https://scholar.google.com",  # Some sites require a referrer
    }

    # got most of the patterns from here from reverse engineering the unpaywall chrome extension
    unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email=unpaywall@impactstory.org"

    try:
        pdf_url = None

        # --- Unpaywall Check ---
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(unpaywall_url)
            response.raise_for_status()
            data = response.json()

            paper_url = data.get("doi_url")

            if data.get("is_oa"):
                logger.info(f"Paper is Open Access according to Unpaywall. DOI: {doi}")

                if data.get("best_oa_location") and data["best_oa_location"].get("url_for_pdf"):
                    logger.info(f"Found direct PDF URL from Unpaywall: {data['best_oa_location']['url_for_pdf']}")
                    pdf_url = data["best_oa_location"]["url_for_pdf"]  # Return directly if available
                    return pdf_url

            else:
                logger.info(f"Paper is NOT Open Access according to Unpaywall. DOI: {doi}")

        # Get final redirected URL (important for DOI links)
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            response = await client.get(paper_url, headers=headers)
            response.raise_for_status()
            logger.info(f"Final URL after redirect: {response.url}")

        final_url = str(response.url)

        # async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        #     response = await client.get(final_url, headers=headers)
        #     response.raise_for_status()

        selector = Selector(text=response.text)

        # --- Meta Tag Check ---
        meta_pdf_url = selector.xpath("//meta[@name='citation_pdf_url']/@content").get()
        if meta_pdf_url:
            logger.info(f"Found PDF URL in meta tag: {meta_pdf_url}")
            return meta_pdf_url

        # --- Domain-Specific Link Checks ---
        for link in selector.xpath("//a"):
            href = link.xpath("@href").get()
            if not href:
                continue

            # 1. Nature.com (Pattern 1)
            if "nature.com" in final_url:
                match = re.search(r"/nature/journal/.+?/pdf/(.+?)\.pdf$", href)
                if match:
                    pdf_url = httpx.URL(final_url).join(href).unicode_string()
                    logger.info(f"Found PDF URL (Nature.com Pattern 1): {pdf_url}")
                    return pdf_url

                # 2. Nature.com (Pattern 2)
                match = re.search(r"/articles/nmicrobiol\d+\.pdf$", href)
                if match:
                    pdf_url = httpx.URL(final_url).join(href).unicode_string()
                    logger.info(f"Found PDF URL (Nature.com Pattern 2): {pdf_url}")
                    return pdf_url

            # 3. NEJM
            if "nejm.org" in final_url:
                if link.xpath("@data-download-content").get() == "Article":
                    pdf_url = httpx.URL(final_url).join(href).unicode_string()
                    logger.info(f"Found PDF URL (NEJM): {pdf_url}")
                    return pdf_url

            # 4. Taylor & Francis Online
            if "tandfonline.com" in final_url:
                match = re.search(r"/doi/pdf/10.+?needAccess=true", href, re.IGNORECASE)
                if match:
                    pdf_url = httpx.URL(final_url).join(href).unicode_string()
                    logger.info(f"Found PDF URL (Taylor & Francis): {pdf_url}")
                    return pdf_url

            # 5. Centers for Disease Control (CDC)
            if "cdc.gov" in final_url:
                if "noDecoration" == link.xpath("@class").get() and re.search(r"\.pdf$", href):
                    pdf_url = httpx.URL(final_url).join(href).unicode_string()
                    logger.info(f"Found PDF URL (CDC): {pdf_url}")
                    return pdf_url

            # 6. ScienceDirect
            if "sciencedirect.com" in final_url:
                pdf_url_attribute = link.xpath("@pdfurl").get()
                if pdf_url_attribute:
                    pdf_url = httpx.URL(final_url).join(pdf_url_attribute).unicode_string()
                    logger.info(f"Found PDF URL (ScienceDirect): {pdf_url}")
                    return pdf_url

        # 7. IEEE Explore (check within the entire page content)
        if "ieeexplore.ieee.org" in final_url:
            match = re.search(r'"pdfPath":"(.+?)\.pdf"', response.text)
            if match:
                pdf_path = match.group(1) + ".pdf"
                pdf_url = "https://ieeexplore.ieee.org" + pdf_path
                logger.info(f"Found PDF URL (IEEE Explore): {pdf_url}")
                return pdf_url

        # --- General PDF Pattern Check (Fallback) ---
        # use the last 3 characters of the DOI to match the link because it's a commmon pattern
        # for it to be included in the URL. This is to avoid false positives.
        # Not always the case though.
        doi_last_3 = doi[-3:] if len(doi) >= 3 else doi
        PDF_PATTERNS = [
            ".pdf",
            "/pdf/",
            "pdf/",
            "download",
            "fulltext",
            "article",
            "viewer",
            "content/pdf",
            "/nature/journal",
            "/articles/",
            "/doi/pdf/",
        ]
        pdf_links = selector.css("a::attr(href)").getall()  # get all links here to loop through

        for link in pdf_links:  # loop through
            if any(pattern in link.lower() for pattern in PDF_PATTERNS):
                # check if any of the patterns are in the link and the doi_last_3 is in the link
                if doi_last_3 in link.lower():
                    pdf_url = httpx.URL(final_url).join(link).unicode_string()
                    logger.info(f"Found PDF link (General Pattern): {pdf_url}")
                    return str(pdf_url)

                # if the doi_last_3 is not in the link, check if the link is a pdf, do this as final.
                pdf_url = httpx.URL(final_url).join(link).unicode_string()
                logger.info(f"Found PDF link (General Pattern): {pdf_url}")
                return str(pdf_url)

        logger.warning("No PDF link found on the page.")
        return None

    except httpx.HTTPStatusError as e:
        logger.error(f"Unpaywall API error ({e.response.status_code}): {e}")
        if e.response.status_code == 404:
            logger.error(f"Paper with DOI {doi} not found by Unpaywall")
        return None

    except httpx.RequestError as e:
        logger.error(f"Request error: {e}")
        return None

    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")
        return None


async def download_paper(doi: str, title: str, output_dir: str = "downloads") -> Optional[str]:
    """
    Downloads a paper PDF given its DOI and title.
    If Unpaywall fails, it scrapes the article page to find the PDF.

    Args:
        doi: The DOI of the paper.
        title: The title of the paper (for the filename).
        output_dir: The directory to save the downloaded PDF.

    Returns:
        The file path of the downloaded PDF if successful, otherwise None.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Check Unpaywall and scrape if necessary
        # pdf_url = await find_pdf_url(doi)

        pdf_link = await scrape_pdf_link(doi)

        print("{pdf_link}", "{pdf_url}")

        if not pdf_link:
            logger.error(f"Could not find a PDF link for DOI: {doi}")
            return None

        # Sanitize title for a safe filename
        # safe_title = "".join(c if c.isalnum() or c in "._-" else "_" for c in title)
        safe_title = re.sub(r"[^\w\-_\.]", "_", title)
        file_name = f"{safe_title}_{doi.replace('/', '_')}.pdf"
        file_path = os.path.join(output_dir, file_name)

        # Check if file already exists
        if os.path.exists(file_path):
            logger.info(f"Skipping download. PDF for DOI: {doi} already exists at {file_path}.")
            return file_path

        # Download the PDF
        async with httpx.AsyncClient() as client:
            try:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36",
                    "Accept": "application/pdf",
                    "Referer": "https://scholar.google.com",  # Some sites require a referrer
                }

                response = await client.get(pdf_link, headers=headers, follow_redirects=True, timeout=30)
                response.raise_for_status()

                content_type = response.headers.get("Content-Type", "")
                logger.info(f"Content-Type received: {content_type}")

                if "pdf" not in content_type.lower():
                    logger.error("The downloaded file is not a PDF!")
                    return None

                with open(file_path, "wb") as f:
                    f.write(response.content)

                logger.info(f"Downloaded PDF for DOI: {doi} to {file_path}")
                return file_path

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error downloading PDF for DOI {doi}: {e.response.status_code}")
                return None
            except httpx.RequestError as e:
                logger.error(f"Request error downloading PDF for DOI {doi}: {e}")
                return None

    except Exception as e:
        logger.exception(f"General error downloading PDF for DOI {doi}: {e}")
        return None


async def download_papers_parallel(results, limit=5, download=False, output_dir="downloads"):
    """
    Displays search results and optionally downloads PDFs in parallel.

    Args:
        results: The search results dictionary.
        limit: The number of top results to display.
        download: If True, download PDFs for the displayed results.
    """
    print(f"\n  Search Results (Top {limit}):")
    doi_title_pairs_for_download = []

    for item in results.get("data", [])[:limit]:
        if download:
            if item.get("externalIds") and item["externalIds"].get("DOI"):
                doi = item["externalIds"]["DOI"]
                title = item.get("title", "Unknown_Title")
                doi_title_pairs_for_download.append((doi, title))  # Collect DOI and title for parallel download
                print(f"      Downloading DOI {doi}")  # Indicate download is initiated
        print()

    downloaded_files = []
    if download and doi_title_pairs_for_download:
        download_tasks = [download_paper(doi, title, output_dir) for doi, title in doi_title_pairs_for_download]
        downloaded_files = await asyncio.gather(*download_tasks)

        successful_downloads = 0
        failed_downloads = 0
        for file_path in downloaded_files:
            if file_path:
                successful_downloads += 1
            else:
                failed_downloads += 1

        logger.info("\n--- Parallel Download Statistics ---")
        logger.info(f"Total papers attempted: {len(doi_title_pairs_for_download)}")
        logger.info(f"Successfully downloaded: {successful_downloads}")
        logger.info(f"Failed downloads: {failed_downloads}")
        logger.info("-----------------------------------\n")

        for i, item in enumerate(results.get("data", [])[:limit]):
            if item.get("externalIds") and item["externalIds"].get("DOI"):
                doi = item["externalIds"]["DOI"]
                file_path = downloaded_files[i]  # Get corresponding result
                if file_path:
                    print(f"      Downloaded DOI: {doi}   PDF: {file_path}")
                else:
                    print(f"      Failed to download PDF for DOI: {doi}")
    return downloaded_files


async def _fetch_paper_details(paper_id: str, fields: List[str]):
    """Helper function to fetch paper details, with caching."""
    if paper_id in paper_cache:
        logger.info(f"Fetching paper details from cache: {paper_id}")
        return paper_cache[paper_id]

    data = await paper_details(paper_id=paper_id, fields=fields)
    if "error" not in data:  # Only cache successful responses
        paper_cache[paper_id] = data
    return data


async def _fetch_citations(paper_id: str, limit: int):
    #  No Caching for citations/references/recommendations (for now) -
    #  they might change more frequently than basic paper details.
    return await paper_citations(paper_id=paper_id, fields=["title"], limit=limit)


async def _fetch_references(paper_id: str, limit: int):
    return await paper_references(paper_id=paper_id, fields=["title"], limit=limit)


async def _fetch_recommendations(paper_id: str, limit: int):
    return await get_paper_recommendations_single(paper_id=paper_id, fields=["title"], limit=limit)


async def process_paper_id(paper_id: str, detail_level: str = "complete", download: bool = False, limit: int = 5):
    """Processes a single paper ID, fetching details and related information."""

    print(f"\nProcessing Paper ID: {paper_id}")

    fields = []
    if detail_level == "basic":
        fields = PaperFields.DEFAULT
    elif detail_level == "detailed":
        fields = PaperFields.DETAILED
    elif detail_level == "complete":
        fields = PaperDetailFields.COMPLETE
    else:
        fields = PaperFields.DEFAULT

    try:
        (
            paper_data,
            citations_data,
            references_data,
            recommendations_data,
        ) = await asyncio.gather(
            _fetch_paper_details(paper_id, fields),
            _fetch_citations(paper_id, limit),
            _fetch_references(paper_id, limit),
            _fetch_recommendations(paper_id, limit),
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # --- (Rest of the printing/processing logic remains largely the same,
    # ---  but uses the results from the helper functions) ---
    if "error" in paper_data:
        print(f"  Error fetching paper details: {paper_data['error']}")
        return  # Exit if basic details fail
    if "error" in citations_data:
        print(f"  Error fetching citations: {citations_data['error']}")
    if "error" in references_data:
        print(f"  Error fetching references: {references_data['error']}")
    if "error" in recommendations_data:
        print(f"  Error fetching recommendations: {recommendations_data['error']}")

    print(f"  Title: {paper_data.get('title', 'N/A')}")
    print(f"  Abstract: {paper_data.get('abstract', 'N/A')[:200]}...")
    print(f"  Year: {paper_data.get('year', 'N/A')}")
    print(f"  DOI: {paper_data['externalIds'].get('DOI', 'N/A')}")
    print(f"  URL: {paper_data.get('url', 'N/A')}")

    # print(paper_data)

    if "authors" in paper_data:
        print("\n  Authors:")
        for author in paper_data["authors"]:
            print(f"    - {author.get('name', 'N/A')} (ID: {author.get('authorId', 'N/A')})")

    print("\n  Citations:")
    for citation in citations_data.get("data", []):
        print(f"    - {citation.get('title', 'N/A')}")

    print("\n  References:")
    for reference in references_data.get("data", []):
        print(f"    - {reference.get('title', 'N/A')}")

    print("\n  Recommendations:")
    for rec in recommendations_data.get("recommendedPapers", []):
        print(f"    - {rec.get('title', 'N/A')}")

    if download:
        if paper_data.get("externalIds") and paper_data["externalIds"].get("DOI"):
            doi = paper_data["externalIds"]["DOI"]
            title = paper_data.get("title", "Unknown_Title")
            file_path = await download_paper(doi, title)
            if file_path:
                print(f"   Downloaded PDF to: {file_path}")
            else:
                print(f"   Failed to download PDF for DOI: {doi}")
        else:
            print("   Cannot download: DOI not found in paper details.")


async def _fetch_author_details(author_id: str, fields: List[str]):
    """Helper function to fetch author details, with caching."""
    if author_id in author_cache:
        logger.info(f"Fetching author details from cache: {author_id}")
        return author_cache[author_id]

    data = await author_details(author_id=author_id, fields=fields)
    if "error" not in data:
        author_cache[author_id] = data
    return data


async def _fetch_author_papers(author_id: str, fields: List[str], limit: int):
    #  Could add caching here later if needed, similar to _fetch_paper_details
    return await author_papers(author_id=author_id, fields=fields, limit=limit)


async def get_detailed_paper_info(paper_id: str):
    """Fetches and prints detailed info for a single paper, including citations and references."""
    try:
        details, citations, references = await asyncio.gather(
            paper_details(paper_id, fields=PaperFields.DETAILED),
            paper_citations(paper_id, fields=CitationReferenceFields.COMPLETE),
            paper_references(paper_id, fields=CitationReferenceFields.COMPLETE),
        )

        if "error" in details:
            print(f"Error fetching details for {paper_id}: {details['error']}")
            return None
        if "error" in citations:
            print(f"Error fetching citations for {paper_id}: {citations['error']}")
            return None
        if "error" in references:
            print(f"Error fetching references for {paper_id}: {references['error']}")
            return None

        # --- Paper Details ---
        print(f"\n--- Paper Details (ID: {paper_id}) ---")
        print(f"  Title: {details.get('title', 'N/A')}")
        print(f"  Abstract: {details.get('abstract', 'N/A')}")
        print(f"  Year: {details.get('year', 'N/A')}")
        # print(f"  Venue: {details.get('venue', 'N/A')}")
        print(f"  DOI: {details.get('externalIds', {}).get('DOI', 'N/A')}")  # Access DOI safely
        print(f"  URL: {details.get('url', 'N/A')}")
        # print(f"  Citation Count: {details.get('citationCount', 'N/A')}")
        # print(f"  Influential Citation Count: {details.get('influentialCitationCount', 'N/A')}")
        # print(f"  Open Access PDF: {details.get('openAccessPdf', 'N/A')}")  # added
        # print(f"  Fields of Study: {', '.join(details.get('fieldsOfStudy', [])) or 'N/A'}")  # added

        # if "authors" in details:
        #     print("  Authors:")
        #     for author in details["authors"]:
        #         print(f"    - {author.get('name', 'N/A')} (ID: {author.get('authorId', 'N/A')})")

        # --- Citations ---
        print(f"\n--- Citations (Total: {len(citations.get('data', []))}) ---")
        # for citation in citations.get("data", []):
        #     print(f"  - Citing Paper Title: {citation.get('title', 'N/A')}")
        #     if "authors" in citation:
        #         authors_str = ", ".join([a.get("name", "Unknown Author") for a in citation["authors"]])
        #         print(f"    Authors: {authors_str}")
        #     print(f"    Year: {citation.get('year', 'N/A')}")
        #     print(f"    Venue: {citation.get('venue', 'N/A')}")
        #     print(f"    Is Influential: {citation.get('isInfluential', 'N/A')}")
        #     if citation.get("contexts"):  # Check if 'contexts' exists (it might not)
        #         print(f"    Contexts:")
        #         for context in citation["contexts"]:
        #             print(f"      - {context}")
        #     if citation.get("intents"):  # Check if 'intents' exists
        #         print(f"    Intents: {', '.join(citation.get('intents', []))}")
        #     print(f"    Paper ID: {citation.get('paperId')}")

        # # --- References ---
        print(f"\n--- References (Total: {len(references.get('data', []))}) ---")
        # for reference in references.get("data", []):
        #     print(f"  - Reference Paper Title: {reference.get('title', 'N/A')}")
        #     if "authors" in reference:
        #         authors_str = ", ".join([a.get("name", "Unknown Author") for a in reference["authors"]])
        #         print(f"    Authors: {authors_str}")
        #     print(f"    Year: {reference.get('year', 'N/A')}")
        #     print(f"    Venue: {reference.get('venue', 'N/A')}")
        #     print(f"    Is Influential: {reference.get('isInfluential', 'N/A')}")

        #     if reference.get("contexts"):
        #         print(f"    Contexts:")
        #         for context in reference["contexts"]:
        #             print(f"      - {context}")
        #     if reference.get("intents"):
        #         print(f"    Intents: {', '.join(reference.get('intents', []))}")
        #     print(f"    Paper ID: {reference.get('paperId')}")

        return details  # Or return a combined data structure (if needed)

    except Exception as e:
        print(f"Error in get_detailed_paper_info for {paper_id}: {e}")
        traceback.print_exc()  # Print the full traceback
        return None


async def process_author_id(author_id: str, detail_level: str = "basic", download: bool = False, limit: int = 5):
    """Processes a single author ID, fetching details and their papers."""

    print(f"\nProcessing Author ID: {author_id}")

    fields = []
    if detail_level == "basic":
        fields = AuthorDetailFields.BASIC
    elif detail_level == "detailed":
        fields = AuthorDetailFields.BASIC + AuthorDetailFields.METRICS
    elif detail_level == "complete":
        fields = AuthorDetailFields.COMPLETE
    else:  # Default
        fields = AuthorDetailFields.BASIC

    try:
        author_data, papers_data = await asyncio.gather(
            _fetch_author_details(author_id, fields),
            _fetch_author_papers(author_id, ["title", "year", "externalIds"], limit),
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    if "error" in author_data:
        print(f"  Error fetching author details: {author_data['error']}")
        return
    if "error" in papers_data:
        print(f"  Error fetching author's papers: {papers_data['error']}")

    print(f"  Name: {author_data.get('name', 'N/A')}")
    print(f"  Affiliations: {', '.join(author_data.get('affiliations', [])) or 'N/A'}")
    print(f"  Citation Count: {author_data.get('citationCount', 'N/A')}")
    print(f"  hIndex: {author_data.get('hIndex', 'N/A')}")

    print(f"\n  Papers (Top {limit}):")
    for paper in papers_data.get("data", []):
        print(f"    - {paper.get('title', 'N/A')} ({paper.get('year', 'N/A')})")
        download_papers_parallel(papers_data, limit=limit, download=download)
        # if download:
        #     if paper.get("externalIds") and paper["externalIds"].get("DOI"):
        #         doi = paper["externalIds"]["DOI"]
        #         title = paper.get("title", "Unknown_Title")
        #         file_path = await download_paper(doi, title)
        #         if file_path:
        #             print(f"      Downloaded PDF to: {file_path}")
        #         else:
        #             print(f"      Failed to download PDF for DOI: {doi}")
        #     else:
        #         print("      Cannot download: DOI not found in paper details.")


async def process_search_query(
    query: str,
    search_type: str = "relevance",
    detail_level: str = "basic",
    download: bool = False,
    limit: int = 5,
):
    """Processes a search query, displaying results and offering further actions."""

    print(f"\nSearching for: '{query}' (Type: {search_type})")
    fields = []
    if detail_level == "basic":
        fields = PaperFields.DEFAULT + ["externalIds"]
    elif detail_level == "detailed":
        fields = PaperFields.DETAILED + ["externalIds"]
    elif detail_level == "complete":
        fields = PaperDetailFields.COMPLETE + ["externalIds"]
    else:
        fields = PaperFields.DEFAULT + ["externalIds"]

    # Create a cache key that includes the query, search type, fields, and limit
    cache_key = (query, search_type, tuple(fields), limit)  # Tuples are hashable

    if cache_key in search_cache:
        print(f"  Fetching search results from cache: {cache_key}")
        results = search_cache[cache_key]
    else:
        if search_type == "relevance":
            results = await paper_relevance_search(query=query, fields=fields, limit=limit)
        elif search_type == "title":
            results = await paper_title_search(query=query, fields=fields)
            if "error" in results:
                print(f"Error during title search {results['error']}")
                return
            if results:
                results = {"data": [results]}
            else:
                results = {"data": []}
        elif search_type == "bulk":
            results = await process_search_query_bulk(query=query, download=download, limit=limit, sort_by="year-desc")
        elif search_type == "author":
            results = await author_search(query=query, fields=["name", "affiliations"], limit=limit)
            if "error" in results:
                print(f"  Error during author search: {results['error']}")
                return
            print(f"\n  Author Search Results (Top {limit}):")
            for author in results.get("data", []):
                print(f"    - {author.get('name', 'N/A')} (ID: {author.get('authorId', 'N/A')})")
                print(f"      Affiliations: {', '.join(author.get('affiliations', [])) or 'N/A'}")
            return  # Stop here for author search
        else:
            print(f"  Invalid search type: {search_type}")
            return

        if "error" in results:
            print(f"  Error during search: {results['error']}")
            return

        # Cache the results *only* if there was no error
        search_cache[cache_key] = results

    print(f"\n  Search Results (Top {limit}):")
    for item in results.get("data", [])[:limit]:
        print(f"    - {item.get('title', 'N/A')}")
        if "paperId" in item:
            print(f"      Paper ID: {item.get('paperId')}")
        # if "authors" in item:
        #     author_names = [author.get("name", "Unknown Author") for author in item.get("authors", [])]
        #     print(f"      Authors: {', '.join(author_names)}")

        print(f"      Year: {item.get('year', 'N/A')}")
        print(f"      DOI: {item['externalIds'].get('DOI', 'N/A')}")
        print(f"      URL: {item.get('url', 'N/A')}")

        download_papers_parallel(results, limit=limit, download=download)

        # if download:
        #     if item.get("externalIds") and item["externalIds"].get("DOI"):
        #         doi = item["externalIds"]["DOI"]
        #         title = item.get("title", "Unknown_Title")
        #         file_path = await download_paper(doi, title)
        #         if file_path:
        #             print(f"      Downloaded PDF to: {file_path}")
        #         else:
        #             print(f"      Failed to download PDF for DOI: {doi}")
        #     else:
        #         print("      Cannot download: DOI not found in paper details.")
        # print()


async def process_search_query_bulk(
    query: str,
    download: bool = False,
    limit: int = 5,
    sort_by: str = None,
):
    """Processes a search query using the bulk search endpoint."""

    print(f"\nSearching for: '{query}'")

    # --- Use BULK_SEARCH fields ---
    fields = PaperFields.BULK_SEARCH

    sort_param = None
    if sort_by == "year":
        sort_param = "publicationDate:asc"  # Oldest first
    elif sort_by == "year-desc":
        sort_param = "publicationDate:desc"  # Newest first
    elif sort_by == "citationCount":
        sort_param = "citationCount:desc"  # Most cited first
    elif sort_by == "citationCount-asc":
        sort_param = "citationCount:asc"
    elif sort_by == "paperId":
        sort_param = "paperId:asc"
    elif sort_by == "paperId-desc":
        sort_param = "paperId:desc"

    # results = await paper_bulk_search(query=query, fields=fields, sort=sort_param)

    # if "error" in results:
    #     print(f"  Error during search: {results['error']}")
    #     return
    # --- After getting initial search results ---
    # paper_ids = [item["paperId"] for item in results["data"]]

    # Example: Fetch detailed info for the top 3 papers:
    # for paper_id in paper_ids[:limit]:  # Limit to 3 for demonstration
    #     await get_detailed_paper_info(paper_id)

    all_results = []
    token = None  # Initialize token to None
    remaining = limit  # Initialize remaining

    while True:  # Loop for pagination
        results = await paper_bulk_search(query=query, fields=fields, sort=sort_param, token=token)

        if "error" in results:
            print(f"  Error during search: {results['error']}")
            return

        if "data" in results:
            all_results.extend(results["data"])  # Accumulate results

        remaining -= len(results.get("data", []))  # Update the remaining items to retrieve
        token = results.get("token")  # Get the continuation token

        # Break the loop if:
        # 1. We have enough results OR
        # 2. There's no more data (no token)
        # The condition to break the for loop
        if remaining <= 0 or not token:
            break

        print(f"  Retrieved {len(all_results)} results, fetching more...")  # Added a print to see the progress

    print(f"\n  Search Results (Top {limit}):")

    # Fetch detailed info and print, *only* for the desired number of results
    for item in all_results[:limit]:
        await get_detailed_paper_info(item["paperId"])

        # for item in results.get("data", []):
        #     print(f"    - {item.get('title', 'N/A')}")
        #     print(f"      Paper ID: {item.get('paperId')}")  # Always print Paper ID
        #     author_names = [author.get("name", "Unknown Author") for author in item.get("authors", [])]
        #     print(f"      Authors: {', '.join(author_names)}")
        #     print(f"      Year: {item.get('year', 'N/A')}")
        #     print(f"      URL: {item.get('url', 'N/A')}")

        # if download:
        #     if item.get("externalIds") and item["externalIds"].get("DOI"):
        #         doi = item["externalIds"]["DOI"]
        #         title = item.get("title", "Unknown_Title")
        #         file_path = await download_paper(doi, title)
        #         if file_path:
        #             print(f"      Downloaded PDF to: {file_path}")
        #         else:
        #             print(f"      Failed to download PDF for DOI: {doi}")
        #     else:
        #         print("      Cannot download: DOI not found in paper details.")
    return results  # Return the final results (for further processing)


def usage():
    """Prints detailed usage instructions, including argument descriptions,
    examples, environment variable explanations, and search type details.
    Uses colorama for colored output.
    """

    print(f"\n{Style.BRIGHT}Semantic Scholar Research Tool{Style.RESET_ALL}")
    print("-----------------------------------")
    print("A command-line tool to interact with the Semantic Scholar API, allowing you to:")
    print("  - Retrieve details about papers and authors.")
    print("  - Perform various types of searches.")
    print("  - Download PDFs (when available).")

    print(f"\n{Style.BRIGHT}Usage:{Style.RESET_ALL}")
    print("  python semantic-scholar-research.py <type> <id> [options]")

    print(f"\n{Style.BRIGHT}Arguments:{Style.RESET_ALL}")
    print(f"  {Style.BRIGHT}<type>{Style.RESET_ALL}  (Required) The type of operation to perform:")
    print(f"    - {Style.BRIGHT}paper{Style.RESET_ALL}:  Retrieve details about a specific paper.")
    print(f"    - {Style.BRIGHT}author{Style.RESET_ALL}: Retrieve details about a specific author.")
    print(f"    - {Style.BRIGHT}search{Style.RESET_ALL}: Perform a search for papers or authors.")

    print(f"\n  {Style.BRIGHT}<id>{Style.RESET_ALL}    (Required) The identifier for the operation:")
    print(f"    - For {Style.BRIGHT}paper{Style.RESET_ALL} type:  A paper identifier (see accepted formats below).")
    print(f"    - For {Style.BRIGHT}author{Style.RESET_ALL} type: A Semantic Scholar Author ID (e.g., '1741101').")
    print(f"    - For {Style.BRIGHT}search{Style.RESET_ALL} type: The search query string (e.g., 'quantum computing').")

    print(f"\n{Style.BRIGHT}Options:{Style.RESET_ALL}")
    print(f"  {Style.BRIGHT}-s, --search_type{Style.RESET_ALL}  (Only for 'search' type)")
    print("    Specifies the type of search to perform.  Defaults to 'relevance'.")
    print(f"    - {Style.BRIGHT}relevance{Style.RESET_ALL}:   General keyword search, sorted by relevance.")
    print(f"    - {Style.BRIGHT}title{Style.RESET_ALL}:       Search for a paper by its exact title.")
    print(f"    - {Style.BRIGHT}bulk{Style.RESET_ALL}:        Bulk search for papers, suitable for larger result sets.")
    print(f"    - {Style.BRIGHT}author{Style.RESET_ALL}:      Search for authors by name.")

    print(f"\n  {Style.BRIGHT}-d, --detail_level{Style.RESET_ALL}")
    print("    Controls the amount of information retrieved. Defaults to 'basic'.")
    print(
        f"    - {Style.BRIGHT}basic{Style.RESET_ALL}:      Returns essential information (title, abstract, year, authors, URL)."
    )
    print(
        f"    - {Style.BRIGHT}detailed{Style.RESET_ALL}:    Includes basic details plus references, citations, venue, and influential citation count."
    )
    print(
        f"    - {Style.BRIGHT}complete{Style.RESET_ALL}:   Returns the most comprehensive data, including all fields from 'detailed' plus additional fields like publication venue details and fields of study."
    )
    print(f"  {Style.BRIGHT}-dl, --download{Style.RESET_ALL}")
    print("      Attempts to download PDFs for the retrieved papers (if available).")

    print(f"\n  {Style.BRIGHT}-l, --limit{Style.RESET_ALL}  (For 'search' and 'author' types)")
    print("    Specifies the maximum number of results to return. Defaults to 5.")
    print("    For 'search', this affects the initial search results.")
    print("    For 'author', this limits the number of papers listed for the author.")

    print(f"\n {Style.BRIGHT}-so --sort_by{Style.RESET_ALL} (For 'search' type when using '--search_type bulk')")
    print("  Controls the sorting of search results. Default is year-desc.")
    print("  Available options: ")
    print(f"    - {Style.BRIGHT}year{Style.RESET_ALL}: Sort by publication year, oldest first.")
    print(f"    - {Style.BRIGHT}year-desc{Style.RESET_ALL}: Sort by publication year, newest first.")
    print(f"    - {Style.BRIGHT}citationCount{Style.RESET_ALL}: Sort by citation count, most cited first.")
    print(f"    - {Style.BRIGHT}citationCount-asc{Style.RESET_ALL}: Sort by citation count, least cited first.")
    print(f"    - {Style.BRIGHT}paperId{Style.RESET_ALL}: Sort by Semantic Scholar Paper ID, ascending.")
    print(f"    - {Style.BRIGHT}paperId-desc{Style.RESET_ALL}: Sort by Semantic Scholar Paper ID, descending.")

    print(f"\n  {Style.BRIGHT}-h, --help{Style.RESET_ALL}")
    print("      Displays this help message and exits.")

    print(f"\n{Style.BRIGHT}Paper ID Formats:{Style.RESET_ALL}")
    print("  The 'paper' type accepts various identifier formats:")
    print("    - Semantic Scholar ID:  e.g., '649def34f8be52c8b66281af98ae884c09aef38b'")
    print("    - CorpusId:             e.g., 'CorpusId:215416146'")
    print("    - DOI:                  e.g., 'DOI:10.18653/v1/N18-3011'")
    print("    - ARXIV:                e.g., 'ARXIV:2106.15928'")
    print("    - MAG:                  e.g., 'MAG:112218234'")
    print("    - ACL:                  e.g., 'ACL:W12-3903'")
    print("    - PMID:                 e.g., 'PMID:19872477'")
    print("    - PMCID:                e.g., 'PMCID:2323736'")
    print("    - URL:                  e.g., 'URL:https://arxiv.org/abs/2106.15928v1'")
    print("      (Supported URL domains: semanticscholar.org, arxiv.org, aclweb.org, acm.org, biorxiv.org)")

    print(f"\n{Style.BRIGHT}Environment Variables:{Style.RESET_ALL}")
    print(f"  {Style.BRIGHT}SEMANTIC_SCHOLAR_API_KEY{Style.RESET_ALL} (Optional):")
    print("    Your Semantic Scholar API key.  Provides higher rate limits and access to some")
    print("    features that require authentication.  Obtain a key from the Semantic Scholar website.")
    print("    If not set, the script will use unauthenticated access (with lower rate limits).")

    print(f"\n{Style.BRIGHT}Examples:{Style.RESET_ALL}")
    print("  1. Get basic details for a paper by DOI:")
    print(f"     {Fore.GREEN}python semantic-scholar-research.py paper DOI:10.18653/v1/N18-3011{Style.RESET_ALL}")

    print("\n  2. Get complete details for a paper by Semantic Scholar ID:")
    print(
        f"     {Fore.GREEN}python semantic-scholar-research.py paper 649def34f8be52c8b66281af98ae884c09aef38b -d complete{Style.RESET_ALL}"
    )

    print("\n  3. Get basic details for an author:")
    print(f"     {Fore.GREEN}python semantic-scholar-research.py author 1741101{Style.RESET_ALL}")

    print("\n  4. Search for papers related to 'quantum computing' (relevance search):")
    print(f"     {Fore.GREEN}python semantic-scholar-research.py search 'quantum computing'{Style.RESET_ALL}")

    print("\n  5. Search for a paper by its exact title:")
    print(f"     {Fore.GREEN}python semantic-scholar-research.py search 'Attention is all you need' -s title{Style.RESET_ALL}")

    print("\n  6. Perform a bulk search for 'machine learning', sorting by year (oldest first):")
    print(f"     {Fore.GREEN}python semantic-scholar-research.py search 'machine learning' -s bulk -so year{Style.RESET_ALL}")

    print("\n  7. Search for papers related to 'deep learning' and download PDFs:")
    print(f"     {Fore.GREEN}python semantic-scholar-research.py search 'deep learning' -dl -l 10{Style.RESET_ALL}")

    print("\n  8. Search for authors named 'Yoshua Bengio':")
    print(f"     {Fore.GREEN}python semantic-scholar-research.py search 'Yoshua Bengio' -s author{Style.RESET_ALL}")

    print("\n  9. Get detailed information about an author and list their top 10 papers:")
    print(f"     {Fore.GREEN}python semantic-scholar-research.py author 1741101 -d detailed -l 10{Style.RESET_ALL}")
    print("\n")


async def main():
    """Main function to handle command-line arguments and orchestrate the workflow."""
    parser = argparse.ArgumentParser(description="Semantic Scholar API Client", add_help=False)

    # Define command-line arguments
    parser.add_argument(
        "type",
        choices=["paper", "author", "search"],
        nargs="?",
        help="Type of operation: 'paper' for paper details, 'author' for author details, 'search' for searching.",
    )
    parser.add_argument("id", nargs="?", help="Identifier (Paper ID, Author ID, or Search Query).")

    parser.add_argument(
        "-s",
        "--search_type",
        choices=["relevance", "title", "bulk", "author"],
        default="relevance",
        help="Type of search (only for 'search' type).",
    )
    parser.add_argument(
        "-d",
        "--detail_level",
        choices=["basic", "detailed", "complete"],
        default="basic",
        help="Set the detail level for any given operation",
    )
    parser.add_argument("-dl", "--download", action="store_true", help="Download available PDFs.")
    parser.add_argument("-l", "--limit", type=int, default=5, help="Maximum number of search results. Defaults to 5.")
    parser.add_argument(
        "-so",
        "--sort_by",
        type=str,
        default="year-desc",
        choices=[
            "year",
            "year-desc",
            "citationCount",
            "citationCount-asc",
            "paperId",
            "paperId-desc",
        ],
        help="Sort results. Options: year, year-desc, citationCount, citationCount-asc, paperId, paperId-desc",
    )
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    # Check if no arguments or only -h/--help is provided
    if len(sys.argv) <= 1 or (len(sys.argv) == 2 and (sys.argv[1] == "-h" or sys.argv[1] == "--help")):
        usage()  # Show usage and exit
        return

    args = parser.parse_args()

    await initialize_client()  # Initialize the HTTP client

    try:
        if args.type == "paper":
            await process_paper_id(args.id, args.detail_level, args.download)  # No limit for paper_id
        elif args.type == "author":
            await process_author_id(args.id, args.detail_level, args.download, args.limit)  # Pass limit
        elif args.type == "search":
            await process_search_query(args.id, args.search_type, args.detail_level, args.download, args.limit)  # Pass limit
        # If invalid arguments passed
        else:
            usage()

    finally:
        await cleanup_client()  # Clean up the HTTP client


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        traceback.print_exc()  # More detailed traceback
        print(f"An unexpected error occurred: {e}")
