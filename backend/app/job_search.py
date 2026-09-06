import os
import json
import time

from urllib.parse import quote
from urllib.request import Request, urlopen

from dotenv import load_dotenv


load_dotenv()

JINA_API_KEY = os.getenv("JINA_API_KEY")

if not JINA_API_KEY:
    raise ValueError(
        "JINA_API_KEY is not configured in .env"
    )


JINA_SEARCH_URL = "https://s.jina.ai/"
JINA_READER_URL = "https://r.jina.ai/"


# ---------------------------------
# Detect Search / Listing Pages
# ---------------------------------

def is_likely_job_page(
    title: str,
    url: str
) -> bool:

    text = f"{title} {url}".lower()

    blocked_phrases = [
        "just a moment",
        "verify you are human",
        "access denied",
        "403 forbidden",

        "jobs in pakistan",
        "jobs in lahore",
        "jobs in karachi",
        "jobs in islamabad",

        "find jobs",
        "job search",
        "job-search",
        "job listings",
        "job listing",
        "open roles",
        "jobs page",
        "job results",

        "/jsearch/",
        "/jobs?",
        "/search?",
        "/job-search/",
        "/job-search?",
        "/jobs/search",
        "/job-search-results",
        "/jobs_c4/",
    ]

    for phrase in blocked_phrases:

        if phrase in text:
            return False

    return True


# ---------------------------------
# Validate Reader Content
# ---------------------------------

def has_job_content(
    text: str
) -> bool:

    if not text:
        return False

    text_lower = text.lower().strip()

    # ---------------------------------
    # Reject verification/block pages
    # ---------------------------------

    blocked_content = [
        "just a moment...",
        "just a moment",
        "verify you are human",
        "security verification",
        "checking your browser",
        "enable javascript and cookies",
        "cloudflare",
        "access denied",
        "403 forbidden",
        "please wait while we verify",
    ]

    for phrase in blocked_content:

        if phrase in text_lower:
            return False


    # ---------------------------------
    # Job-related indicators
    # ---------------------------------

    indicators = [
        "responsibilities",
        "requirements",
        "qualifications",
        "experience",
        "skills",
        "education",
        "job description",
        "apply now",
        "about the role",
        "what you will do",
        "what we're looking for",
        "what we are looking for",
        "employment type",
        "job type",
    ]


    matches = sum(
        1
        for indicator in indicators
        if indicator in text_lower
    )


    # Require multiple job sections
    return matches >= 2


# ---------------------------------
# Jina Search
# ---------------------------------

def search_jobs(
    query: str,
    num_results: int = 5
) -> list[dict]:

    if not query.strip():
        return []


    encoded_query = quote(query)

    url = f"{JINA_SEARCH_URL}?q={encoded_query}"

    last_error = None


    for attempt in range(3):

        try:

            print(
                f"Jina search: '{query}' "
                f"(attempt {attempt + 1})"
            )


            request = Request(
                url,

                headers={
                    "Authorization":
                        f"Bearer {JINA_API_KEY}",

                    "Accept":
                        "application/json",
                },

                method="GET",
            )


            with urlopen(
                request,
                timeout=60
            ) as response:

                response_data = (
                    response
                    .read()
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                )


            data = json.loads(
                response_data
            )


            results = data.get(
                "data",
                []
            )


            jobs = []


            # Look at extra results because
            # many search results are listing pages.
            max_results_to_check = (
                max(
                    num_results * 4,
                    20
                )
            )


            for result in results[
                :max_results_to_check
            ]:

                title = result.get(
                    "title",
                    ""
                ).strip()


                job_url = result.get(
                    "url",
                    ""
                ).strip()


                content = result.get(
                    "content",
                    ""
                )


                description = result.get(
                    "description",
                    ""
                )


                if not title or not job_url:
                    continue


                # ---------------------------------
                # Skip obvious listing pages
                # ---------------------------------

                if not is_likely_job_page(
                    title,
                    job_url
                ):

                    print(
                        f"Skipping listing page: "
                        f"{job_url}"
                    )

                    continue


                job_description = (
                    description
                    or content[:3000]
                )


                jobs.append({

                    "title":
                        title,

                    "url":
                        job_url,

                    "description":
                        job_description,

                    "source":
                        "Jina Search"
                })


                if len(jobs) >= num_results:
                    break


            return jobs


        except Exception as error:

            last_error = error


            print(
                f"Jina search failed: "
                f"{error}"
            )


            if attempt < 2:

                delay = 3 * (
                    attempt + 1
                )


                print(
                    f"Retrying Jina search "
                    f"in {delay} seconds..."
                )


                time.sleep(delay)


    raise RuntimeError(
        "Jina job search failed after "
        f"3 attempts: {last_error}"
    )


# ---------------------------------
# Read Job Page With Jina Reader
# ---------------------------------

def read_job_page(
    job_url: str
) -> str:

    if not job_url.strip():
        return ""


    reader_url = (
        f"{JINA_READER_URL}"
        f"{job_url}"
    )


    for attempt in range(2):

        try:

            print(
                f"Jina Reader: {job_url} "
                f"(attempt {attempt + 1})"
            )


            request = Request(
                reader_url,

                headers={
                    "Authorization":
                        f"Bearer {JINA_API_KEY}",

                    "Accept":
                        "text/plain",
                },

                method="GET",
            )


            with urlopen(
                request,
                timeout=60
            ) as response:

                page_content = (
                    response
                    .read()
                    .decode(
                        "utf-8",
                        errors="ignore"
                    )
                )


            if page_content.strip():

                page_content = (
                    page_content.strip()
                )


                # ---------------------------------
                # Validate actual job content
                # ---------------------------------

                if has_job_content(
                    page_content
                ):

                    print(
                        "Jina Reader found "
                        "useful job content."
                    )


                    return page_content


                print(
                    "Jina Reader page does not "
                    "contain enough job information."
                )


                return ""


        except Exception as error:

            print(
                f"Jina Reader failed: "
                f"{error}"
            )


            if attempt < 1:

                time.sleep(3)


    print(
        f"Jina Reader could not read: "
        f"{job_url}"
    )


    return ""


# ---------------------------------
# Enrich Jobs With Real Page Content
# ---------------------------------

def enrich_jobs_with_reader(
    jobs: list[dict]
) -> list[dict]:

    enriched_jobs = []


    # Maximum 10 Reader requests
    for job in jobs[:10]:

        job_url = job.get(
            "url",
            ""
        )


        original_description = job.get(
            "description",
            ""
        )


        page_content = read_job_page(
            job_url
        )


        if page_content:

            # Keep Gemini prompt manageable
            page_content = (
                page_content[:6000]
            )


            job_description = (
                page_content
            )


            reader_available = True


            print(
                f"Reader content added for: "
                f"{job.get('title', '')}"
            )


        else:

            job_description = (
                original_description
            )


            reader_available = False


            print(
                f"Using search description for: "
                f"{job.get('title', '')}"
            )


        enriched_job = {

            **job,

            "description":
                job_description,

            "reader_content_available":
                reader_available
        }


        enriched_jobs.append(
            enriched_job
        )


    return enriched_jobs


# ---------------------------------
# Search Multiple Queries
# ---------------------------------

def search_jobs_for_queries(
    queries: list[str],
    results_per_query: int = 5
) -> list[dict]:

    all_jobs = []


    for query in queries:

        try:

            jobs = search_jobs(
                query=query,
                num_results=results_per_query
            )


            for job in jobs:

                job["search_query"] = (
                    query
                )


                all_jobs.append(
                    job
                )


        except Exception as error:

            print(
                f"Skipping failed query "
                f"'{query}': {error}"
            )


            continue


    # ---------------------------------
    # Remove Duplicate URLs
    # ---------------------------------

    unique_jobs = []

    seen_urls = set()


    for job in all_jobs:

        job_url = job.get(
            "url",
            ""
        ).strip()


        if not job_url:
            continue


        if job_url in seen_urls:
            continue


        seen_urls.add(
            job_url
        )


        unique_jobs.append(
            job
        )


    print(
        f"Found {len(unique_jobs)} "
        f"unique jobs before Reader."
    )


    # ---------------------------------
    # Read Actual Job Pages
    # ---------------------------------

    enriched_jobs = (
        enrich_jobs_with_reader(
            unique_jobs
        )
    )


    return enriched_jobs