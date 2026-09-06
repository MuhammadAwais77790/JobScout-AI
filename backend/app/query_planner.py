import os
import json
import time

from dotenv import load_dotenv
from google import genai


# ---------------------------------
# Environment
# ---------------------------------

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured in .env"
    )


client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ---------------------------------
# Generate Dynamic Job Queries
# ---------------------------------

def generate_job_queries(
    profile: dict,
    retrieved_context: list | None = None,
    number_of_queries: int = 5
) -> list[str]:

    retrieved_context = (
        retrieved_context or []
    )

    skills = profile.get(
        "skills",
        []
    )

    job_titles = profile.get(
        "job_titles",
        []
    )

    experience = profile.get(
        "experience",
        []
    )

    summary = profile.get(
        "summary",
        ""
    )

    experience_text = "\n".join(
        str(item)
        for item in experience
    )

    rag_context = "\n".join(
        item.get(
            "chunk",
            ""
        )
        for item in retrieved_context
    )


    prompt = f"""
You are an autonomous career search planner.

Your task is to generate job-search
queries for a candidate.

Use ONLY the candidate information
provided below.

Candidate Skills:
{skills}

Candidate Job Titles:
{job_titles}

Candidate Experience:
{experience_text}

Candidate Summary:
{summary}

Relevant CV Context retrieved using RAG:
{rag_context}


IMPORTANT RULES:

1. Generate exactly
   {number_of_queries}
   job-search queries.

2. Queries must be based on the
   candidate's actual skills,
   experience, and possible job roles.

3. Do NOT use a predefined list
   of job queries.

4. Do NOT invent skills or experience.

5. Keep each query concise.

6. Queries should be suitable for
   real job-search websites.

7. Create variations of roles
   where appropriate.

8. Return ONLY a JSON array
   of strings.

Example format:

[
    "SEO Specialist jobs",
    "Content Writer jobs",
    "SEO Content Writer jobs"
]
"""


    models = [
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash"
    ]

    last_error = None


    for model in models:

        for attempt in range(2):

            try:

                print(
                    f"Gemini query planner: "
                    f"model={model}, "
                    f"attempt={attempt + 1}"
                )


                response = (
                    client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={
                            "response_mime_type":
                                "application/json"
                        },
                    )
                )


                queries = json.loads(
                    response.text
                )


                if not isinstance(
                    queries,
                    list
                ):

                    raise ValueError(
                        "Gemini did not return "
                        "a list of job queries."
                    )


                queries = [

                    query.strip()

                    for query in queries

                    if isinstance(
                        query,
                        str
                    )

                    and query.strip()
                ]


                queries = queries[
                    :number_of_queries
                ]


                if not queries:

                    raise ValueError(
                        "No valid job queries "
                        "were generated."
                    )


                print(
                    f"Job queries generated "
                    f"using {model}"
                )


                return queries


            except Exception as error:

                last_error = error

                print(
                    f"Gemini query planner "
                    f"failed using {model}: "
                    f"{error}"
                )


                error_text = str(error)


                if "429" in error_text:

                    print(
                        f"{model} quota unavailable. "
                        f"Trying next model..."
                    )

                    break


                if "503" in error_text:

                    if attempt < 1:

                        print(
                            "Gemini temporarily "
                            "unavailable. Retrying..."
                        )

                        time.sleep(3)

                        continue


                break


    raise RuntimeError(
        "All Gemini query planner "
        f"models failed: {last_error}"
    )