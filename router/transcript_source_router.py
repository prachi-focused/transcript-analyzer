"""Route at graph entry: run LLM on transcript files vs load prior analyses from Postgres."""

from db import list_transcript_ids_in_analyses


def transcript_source_router(state: dict) -> str:
    """
    If ``transcript_analyses`` is empty, go straight to fresh analysis (nothing to reuse).

    Otherwise ask whether to run a new transcript analysis or reuse rows from the DB.

    Returns the next node name.
    """
    try:
        transcript_ids = list_transcript_ids_in_analyses()
    except Exception as e:
        print(
            f"Could not load transcript IDs from transcript_analyses ({e}); "
            "running analyze_transcript."
        )
        return "analyze_transcript"

    if transcript_ids:
        print("--------------------------------")
        print(
            "There is transcript analysis in the database for transcript_ids:\n  "
            + "\n  ".join(transcript_ids)
        )
        print("--------------------------------\n")
        print("Run a new analysis? (yes/no): ", end="")

        raw = input().strip().lower()
        if raw in ("yes", "y"):
            return "analyze_transcript"
        elif raw in ("no", "n"):
            return "load_DB_analysis"
        else:
            print("Invalid answer. Please enter yes or no (or y/n).")
            return "transcript_source_router"
    else:
        print("No transcript analysis in the database. Running analyze_transcript.")
        return "analyze_transcript"
