import os
from typing import List, Dict, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured in .env")

client = genai.Client(api_key=GEMINI_API_KEY)

# Primary model + working fallback
MODELS = [
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
]


class JobAnalysis(BaseModel):
    job_title: str = ""

    required_skills: List[str] = Field(
        default_factory=list
    )

    responsibilities: List[str] = Field(
        default_factory=list
    )

    experience_requirements: List[str] = Field(
        default_factory=list
    )

    education_requirements: List[str] = Field(
        default_factory=list
    )

    # Candidate matching
    match_reason: str = ""

    strengths: List[str] = Field(
        default_factory=list
    )

    gaps: List[str] = Field(
        default_factory=list
    )

    # Semantic job quality
    is_individual_job: bool = True

    is_active_job: bool = True

    quality_reason: str = ""


class JobAnalysisResult(BaseModel):
    job_index: int

    analysis: JobAnalysis


class JobAnalysisResponse(BaseModel):
    results: List[JobAnalysisResult]


def analyze_jobs_with_ai(
    jobs: List[Dict],
    candidate_profile: Optional[Dict] = None,
) -> List[Dict]:
    """
    Analyze all jobs in ONE Gemini request.

    Gemini performs:
    1. Job requirement extraction
    2. Candidate matching
    3. Individual-job quality check
    4. Active/inactive check
    """

    if not jobs:
        return []

    candidate_profile = candidate_profile or {}

    candidate_skills = candidate_profile.get(
        "skills",
        []
    )

    candidate_experience = candidate_profile.get(
        "experience",
        []
    )

    candidate_education = candidate_profile.get(
        "education",
        []
    )

    candidate_job_titles = candidate_profile.get(
        "job_titles",
        []
    )

    candidate_location = candidate_profile.get(
        "location",
        ""
    )

    job_blocks = []

    for index, job in enumerate(jobs):

        title = str(
            job.get("title", "")
        ).strip()

        company = str(
            job.get("company", "")
        ).strip()

        url = str(
            job.get("url", "")
        ).strip()

        description = str(
            job.get("description", "")
        ).strip()

        # Keep the prompt under control.
        description = description[:3000]

        job_blocks.append(
            f"""
JOB INDEX: {index}

TITLE:
{title}

COMPANY:
{company}

URL:
{url}

DESCRIPTION:
{description}
"""
        )

    jobs_text = "\n".join(job_blocks)

    prompt = f"""
You are JobScout AI, an autonomous career-matching
and job-quality agent.

Your task is to analyze EVERY job result against
the candidate profile.

====================================================
CANDIDATE PROFILE
====================================================

Skills:
{candidate_skills}

Experience:
{candidate_experience}

Education:
{candidate_education}

Possible Job Titles:
{candidate_job_titles}

Location:
{candidate_location}


====================================================
JOB RESULTS
====================================================

{jobs_text}


====================================================
A) JOB QUALITY CHECK
====================================================

For every job determine:

is_individual_job

Set TRUE only if the result represents ONE
specific job vacancy/role that a candidate can
reasonably apply for.

Set FALSE for:

- talent directories
- freelancer directories
- candidate directories
- category pages
- search result pages
- job collection pages
- general classifieds pages
- pages listing many candidates
- pages listing many jobs
- company career homepages without a vacancy
- general service pages
- general information pages


is_active_job

Set TRUE when the listing appears currently open.

Set FALSE when the listing explicitly says:

- expired
- closed
- filled
- no longer available
- applications closed
- position closed
- no longer accepting applications


If the status is unclear, do not invent information.

quality_reason

Give ONE short sentence explaining why this
result is a usable individual job or why it should
not be used.


====================================================
B) JOB REQUIREMENTS
====================================================

Extract:

required_skills

responsibilities

experience_requirements

education_requirements


====================================================
C) CANDIDATE MATCH
====================================================

Compare ONLY the candidate profile above with
the job information.

match_reason:

Write 1-3 concise sentences explaining why the
candidate matches or does not match the job.

strengths:

List the candidate skills or experience that
directly support this job.

gaps:

List important requirements that are not supported
by the candidate profile.


====================================================
IMPORTANT RULES
====================================================

1. Never invent candidate information.

2. Never invent job requirements.

3. Do not call a directory a job.

4. Do not call a category page a job.

5. Do not call a search-results page a job.

6. If a listing is explicitly expired or closed,
   is_active_job MUST be false.

7. A high skill match does NOT override bad
   job quality.

8. Return the exact job_index.

9. Keep match_reason concise.

10. Return structured JSON only.
"""

    last_error = None

    for model_name in MODELS:

        try:

            print(
                f"Gemini job analysis: "
                f"{len(jobs)} jobs, "
                f"model={model_name}"
            )

            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                    response_schema=JobAnalysisResponse,
                ),
            )

            if response.parsed:

                parsed = response.parsed

                results = []

                for item in parsed.results:

                    results.append(
                        {
                            "job_index": item.job_index,
                            "analysis": item.analysis.model_dump(),
                        }
                    )

                print(
                    f"Job analysis successful "
                    f"using {model_name}"
                )

                return results

            print(
                f"Gemini returned no parsed response "
                f"using {model_name}"
            )

        except Exception as error:

            last_error = error

            print(
                f"{model_name} unavailable: "
                f"{error}"
            )

    print(
        f"All Gemini job-analysis models failed: "
        f"{last_error}"
    )

    return []


def build_job_analysis_map(
    jobs: List[Dict],
    analyses: List[Dict],
) -> List[Dict]:
    """
    Attach AI analysis to the original jobs.
    """

    analysis_map = {
        item["job_index"]: item["analysis"]
        for item in analyses
        if (
            "job_index" in item
            and "analysis" in item
        )
    }

    updated_jobs = []

    for index, job in enumerate(jobs):

        analysis = analysis_map.get(
            index,
            {
                "job_title": job.get(
                    "title",
                    ""
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
            },
        )

        updated_jobs.append(
            {
                **job,
                "job_analysis": analysis,
            }
        )

    return updated_jobs