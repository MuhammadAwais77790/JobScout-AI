import os
import time

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


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
# Education Schema
# ---------------------------------

class Education(BaseModel):

    degree: str

    institution: str | None = None

    details: str | None = None


# ---------------------------------
# Experience Schema
# ---------------------------------

class Experience(BaseModel):

    job_title: str | None = None

    company: str | None = None

    description: str | None = None


# ---------------------------------
# Language Schema
# ---------------------------------

class Language(BaseModel):

    language: str

    proficiency: str | None = None


# ---------------------------------
# Candidate Profile Schema
# ---------------------------------

class CandidateProfile(BaseModel):

    name: str | None = None

    email: str | None = None

    phone: str | None = None

    location: str | None = None

    skills: list[str]

    education: list[Education]

    experience: list[Experience]

    languages: list[Language]

    job_titles: list[str]

    summary: str | None = None


# ---------------------------------
# Gemini Profile Request
# ---------------------------------

def generate_profile_response(
    prompt: str
):

    # Primary model is lightweight
    # and currently working.
    models = [
        "gemini-3.5-flash-lite",
        "gemini-3.7-flash"
    ]

    last_error = None

    for model in models:

        for attempt in range(2):

            try:

                print(
                    f"Gemini profile request: "
                    f"model={model}, "
                    f"attempt={attempt + 1}"
                )

                response = (
                    client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config={
                            "response_mime_type":
                                "application/json",

                            "response_schema":
                                CandidateProfile,
                        },
                    )
                )

                print(
                    f"Gemini profile successful "
                    f"using {model}"
                )

                return response

            except Exception as error:

                last_error = error

                print(
                    f"Gemini profile failed "
                    f"using {model}: {error}"
                )

                error_text = str(error)

                # ---------------------------------
                # Quota error
                # ---------------------------------

                if "429" in error_text:

                    print(
                        f"{model} quota unavailable. "
                        f"Trying next model..."
                    )

                    break

                # ---------------------------------
                # Temporary server error
                # ---------------------------------

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
        f"All Gemini models failed: "
        f"{last_error}"
    )


# ---------------------------------
# Analyze CV With AI
# ---------------------------------

def analyze_cv_with_ai(
    cv_text: str
) -> dict:

    prompt = f"""
You are an expert CV and recruitment analyzer.

Analyze the CV below and extract the
candidate's information.

IMPORTANT RULES:

1. Extract information ONLY from the CV.

2. Never invent information.

3. Do NOT use a predefined skill list.

4. Extract skills dynamically from the
   candidate's CV.

5. The name must contain ONLY the
   person's name.

6. Never include address, email, phone,
   or other text in the name.

7. Location should contain ONLY a city,
   state/province, or country.

8. Do not put a complete street/home
   address in location.

9. Keep every education record separate.

10. Keep every work experience record
    separate.

11. Identify job titles/roles when
    available.

12. Extract languages and proficiency
    when mentioned.

13. Generate possible job titles ONLY
    from the candidate's actual skills
    and experience.

14. Do not invent companies or job
    experience.

15. If information is unavailable,
    use null or an empty list.

16. Keep the summary short and based
    only on the CV.

CV TEXT:
-------------------------

{cv_text}

-------------------------
"""

    response = generate_profile_response(
        prompt
    )

    profile = (
        CandidateProfile
        .model_validate_json(
            response.text
        )
    )

    return profile.model_dump()