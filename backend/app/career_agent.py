from typing import Dict, List
from urllib.parse import urlparse

from app.cv_parser import extract_cv_text
from app.ai_profile import analyze_cv_with_ai
from app.rag_engine import (
    build_cv_rag,
    retrieve_relevant_chunks,
)
from app.query_planner import generate_job_queries
from app.job_search import search_jobs_for_queries
from app.job_matcher import calculate_match_score

from app.job_analyzer import (
    analyze_jobs_with_ai,
    build_job_analysis_map,
)


BLOCKED_TITLES = {
    "just a moment...",
    "access denied",
    "403 forbidden",
    "verify you are human",
    "security verification",
}


# Social-media pages are not treated as individual job listings.
# Direct job pages from supported job portals remain allowed.
BLOCKED_DOMAINS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "web.facebook.com",
    "instagram.com",
    "www.instagram.com",
    "tiktok.com",
    "www.tiktok.com",
    "pinterest.com",
    "www.pinterest.com",
}


def clean_text(value) -> str:

    if value is None:
        return ""

    return str(value).strip()


def build_fallback_analysis(
    job: Dict,
) -> Dict:

    return {
        "job_title": clean_text(
            job.get("title", "")
        ),

        "required_skills": [],

        "responsibilities": [],

        "experience_requirements": [],

        "education_requirements": [],

        "match_reason": "",

        "strengths": [],

        "gaps": [],

        "is_individual_job": True,

        "is_active_job": True,

        "quality_reason":
            "AI quality assessment was unavailable.",
    }


def get_search_location(
    profile: Dict,
) -> str:

    location = clean_text(
        profile.get(
            "location",
            ""
        )
    )

    if location:
        return location

    return "Pakistan"


def is_quality_job(
    job: Dict,
) -> bool:
    """
    Fast deterministic quality filter.

    Semantic quality is checked later by Gemini
    in the same batch request.
    """

    title = clean_text(
        job.get(
            "title",
            ""
        )
    )

    description = clean_text(
        job.get(
            "description",
            ""
        )
    )

    url = clean_text(
        job.get(
            "url",
            ""
        )
    )

    title_lower = title.lower()

    text_lower = (
        f"{title} {description}"
    ).lower()

    url_lower = url.lower()

    # -------------------------------------------------
    # Social media / non-job platforms
    # -------------------------------------------------

    try:
        hostname = urlparse(url).hostname or ""
        hostname = hostname.lower()
    except Exception:
        hostname = ""

    if hostname in BLOCKED_DOMAINS:
        print(
            "Quality filter removed social-media page: "
            f"{title}"
        )
        return False

    social_markers = [
        "facebook.com",
        "instagram.com",
        "tiktok.com",
        "pinterest.com",
    ]

    if any(
        marker in url_lower
        for marker in social_markers
    ):
        print(
            "Quality filter removed social-media page: "
            f"{title}"
        )
        return False

    # -------------------------------------------------
    # Security / bot pages
    # -------------------------------------------------

    for phrase in BLOCKED_TITLES:

        if (
            phrase in title_lower
            or phrase in text_lower
        ):
            return False

    # -------------------------------------------------
    # Expired / closed pages
    # -------------------------------------------------

    expired_phrases = [

        "job expired",

        "expired job",

        "this job has expired",

        "position has been filled",

        "position is no longer available",

        "no longer accepting applications",

        "applications are closed",

        "job is closed",

        "closed position",
    ]

    if any(
        phrase in text_lower
        for phrase in expired_phrases
    ):
        return False

    # -------------------------------------------------
    # Aggregate / directory pages
    # -------------------------------------------------

    aggregate_phrases = [

        "search results",

        "job search",

        "jobs search",

        "free classifieds",

        "classifieds",

        "job listings",

        "all jobs",

        "latest jobs",

        "browse jobs",

        "find jobs",

        "category",

        "talent directory",

        "freelancer directory",

        "directory page",

    ]

    if any(
        phrase in title_lower
        for phrase in aggregate_phrases
    ):
        return False

    # -------------------------------------------------
    # Search/category URLs
    # -------------------------------------------------

    search_url_markers = [

        "/search",

        "?q=",

        "?query=",

        "/category/",

        "/categories/",

        "/jobs?keyword=",

        "/jobs/search",

        "/search-jobs",
    ]

    if any(
        marker in url_lower
        for marker in search_url_markers
    ):
        return False

    # -------------------------------------------------
    # Generic titles
    # -------------------------------------------------

    generic_titles = {

        "jobs",

        "job",

        "careers",

        "career",

        "vacancies",

        "vacancy",

        "employment",

    }

    if (
        title_lower in generic_titles
        and len(description) < 250
    ):
        return False

    # -------------------------------------------------
    # Empty result
    # -------------------------------------------------

    if (
        not title
        and len(description) < 80
    ):
        return False

    return True


def filter_job_quality(
    jobs: List[Dict],
) -> List[Dict]:

    clean_jobs = []

    for job in jobs:

        if is_quality_job(job):

            clean_jobs.append(job)

        else:

            print(
                "Quality filter removed: "
                f"{clean_text(job.get('title', 'Untitled'))}"
            )

    print(
        f"Quality filter: "
        f"{len(clean_jobs)} of "
        f"{len(jobs)} jobs kept."
    )

    return clean_jobs


def apply_ai_quality_filter(
    analyzed_jobs: List[Dict],
) -> List[Dict]:
    """
    Remove directory, aggregate and inactive
    listings based on the AI semantic assessment.

    No additional Gemini request is made here.
    """

    final_jobs = []

    for job in analyzed_jobs:

        analysis = (
            job.get(
                "job_analysis"
            )
            or {}
        )

        individual = analysis.get(
            "is_individual_job",
            True,
        )

        active = analysis.get(
            "is_active_job",
            True,
        )

        quality_reason = clean_text(
            analysis.get(
                "quality_reason",
                ""
            )
        )

        title = clean_text(
            job.get(
                "title",
                "Untitled"
            )
        )

        # ---------------------------------------------
        # Directory / aggregate
        # ---------------------------------------------

        if not individual:

            print(
                "AI quality filter removed "
                f"directory/aggregate: "
                f"{title} | "
                f"{quality_reason}"
            )

            continue

        # ---------------------------------------------
        # Expired / inactive
        # ---------------------------------------------

        if not active:

            print(
                "AI quality filter removed "
                f"inactive job: "
                f"{title} | "
                f"{quality_reason}"
            )

            continue

        final_jobs.append(job)

    print(
        f"AI quality filter: "
        f"{len(final_jobs)} of "
        f"{len(analyzed_jobs)} jobs kept."
    )

    return final_jobs


def evaluate_jobs(
    jobs: List[Dict],
    candidate_profile: Dict,
) -> List[Dict]:

    if not jobs:
        return []

    candidate_profile = (
        candidate_profile or {}
    )

    candidate_skills = (
        candidate_profile.get(
            "skills",
            []
        )
    )

    # =================================================
    # 1. FAST QUALITY FILTER
    # =================================================

    usable_jobs = filter_job_quality(
        jobs
    )

    if not usable_jobs:
        return []

    # =================================================
    # 2. ONE BATCH AI ANALYSIS
    # =================================================

    try:

        analyses = analyze_jobs_with_ai(
            usable_jobs,
            candidate_profile=candidate_profile,
        )

        analyzed_jobs = build_job_analysis_map(
            jobs=usable_jobs,
            analyses=analyses,
        )

    except Exception as error:

        print(
            f"Job analysis failed, "
            f"using fallback: {error}"
        )

        analyzed_jobs = [

            {
                **job,

                "job_analysis":
                    build_fallback_analysis(
                        job
                    ),
            }

            for job in usable_jobs
        ]

    # =================================================
    # 3. AI SEMANTIC QUALITY FILTER
    # =================================================

    quality_jobs = apply_ai_quality_filter(
        analyzed_jobs
    )

    if not quality_jobs:
        return []

    # =================================================
    # 4. MATCHING + REASONING
    # =================================================

    evaluated_jobs = []

    for job in quality_jobs:

        job_analysis = (
            job.get(
                "job_analysis"
            )
            or {}
        )

        required_skills = (
            job_analysis.get(
                "required_skills",
                []
            )
        )

        match_result = calculate_match_score(
            candidate_skills,
            required_skills,
        )

        evaluation = {

            **match_result,

            "match_reason":
                clean_text(
                    job_analysis.get(
                        "match_reason",
                        ""
                    )
                ),

            "strengths":
                job_analysis.get(
                    "strengths",
                    []
                ),

            "gaps":
                job_analysis.get(
                    "gaps",
                    []
                ),

            "quality_reason":
                clean_text(
                    job_analysis.get(
                        "quality_reason",
                        ""
                    )
                ),

            "is_individual_job":
                job_analysis.get(
                    "is_individual_job",
                    True,
                ),

            "is_active_job":
                job_analysis.get(
                    "is_active_job",
                    True,
                ),
        }

        evaluated_jobs.append(
            {
                **job,

                "evaluation":
                    evaluation,
            }
        )

    # =================================================
    # 5. RANK JOBS
    # =================================================

    evaluated_jobs.sort(
        key=lambda job:
            job.get(
                "evaluation",
                {}
            ).get(
                "match_score",
                0
            ),

        reverse=True,
    )

    # Add rank number.

    for rank, job in enumerate(
        evaluated_jobs,
        start=1,
    ):

        job["rank"] = rank

    return evaluated_jobs


def run_career_agent(
    file_bytes: bytes,
    filename: str,
) -> Dict:

    # =================================================
    # STEP 1 — CV INGEST
    # =================================================

    print(
        "\n========== "
        "STEP 1: CV INGEST "
        "=========="
    )

    cv_text = extract_cv_text(
        file_bytes,
        filename,
    )

    if not cv_text:

        raise ValueError(
            "Could not extract readable text "
            "from the uploaded CV."
        )

    print(
        f"CV text extracted: "
        f"{len(cv_text)} characters"
    )

    # =================================================
    # STEP 2 — AI PROFILE
    # =================================================

    print(
        "\n========== "
        "STEP 2: AI PROFILE "
        "=========="
    )

    profile = analyze_cv_with_ai(
        cv_text
    )

    if not isinstance(
        profile,
        dict
    ):

        profile = {}

    print(
        f"Skills: "
        f"{profile.get('skills', [])}"
    )

    print(
        f"Location: "
        f"{profile.get('location', '')}"
    )

    # =================================================
    # STEP 3 — RAG
    # =================================================

    print(
        "\n========== "
        "STEP 3: RAG "
        "=========="
    )

    rag_data = build_cv_rag(
        cv_text
    )

    retrieved_context = (
        retrieve_relevant_chunks(
            query=(
                "skills experience education "
                "job roles"
            ),

            chunks=rag_data.get(
                "chunks",
                []
            ),

            embeddings=rag_data.get(
                "embeddings",
                []
            ),

            top_k=3,
        )
    )

    # =================================================
    # STEP 4 — DYNAMIC QUERY PLANNING
    # =================================================

    print(
        "\n========== "
        "STEP 4: QUERY PLANNING "
        "=========="
    )

    job_queries = generate_job_queries(
        profile,
        retrieved_context,
        5,
    )

    print(
        f"Generated queries: "
        f"{job_queries}"
    )

    # =================================================
    # STEP 5 — LIVE JOB SEARCH
    # =================================================

    print(
        "\n========== "
        "STEP 5: JOB SEARCH "
        "=========="
    )

    search_location = get_search_location(
        profile
    )

    search_queries = [

        f"{query} {search_location}"

        for query in job_queries

        if clean_text(query)
    ]

    print(
        f"Search location: "
        f"{search_location}"
    )

    jobs = search_jobs_for_queries(
        queries=search_queries,
        results_per_query=3,
    )

    print(
        f"Raw jobs collected: "
        f"{len(jobs)}"
    )

    # =================================================
    # STEP 6 — QUALITY + AI EVALUATION
    # =================================================

    print(
        "\n========== "
        "STEP 6: QUALITY + EVALUATION "
        "=========="
    )

    evaluated_jobs = evaluate_jobs(
        jobs,
        profile,
    )

    print(
        f"Final quality jobs: "
        f"{len(evaluated_jobs)}"
    )

    # =================================================
    # STEP 7 — COMPLETE
    # =================================================

    print(
        "\n========== "
        "STEP 7: COMPLETE "
        "=========="
    )

    return {

        "filename":
            filename,

        "characters_extracted":
            len(cv_text),

        "profile":
            profile,

        "rag": {

            "chunk_count":
                rag_data.get(
                    "chunk_count",
                    0,
                ),

            "retrieved_context":
                retrieved_context,
        },

        "job_queries":
            job_queries,

        "search_location":
            search_location,

        "raw_jobs_count":
            len(jobs),

        "total_jobs":
            len(evaluated_jobs),

        "jobs":
            evaluated_jobs,
    }