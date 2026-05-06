"""Route after transcript analysis: optional policy store update vs end."""

from db import policy_txt_chunks_is_empty


def policy_update_router(state: dict) -> str:
    """
    If the policy vector store has no TXT-ingested chunks, run policy ingest first.

    Otherwise ask whether to update the policy store or skip.

    Returns ``policy_update`` or ``skip_policy_update``.
    """
    try:
        if policy_txt_chunks_is_empty():
            print(
                "No policy_txt chunks in the vector store; running policy_update."
            )
            return "policy_update"
    except Exception as e:
        print(
            f"Could not check policy_chunks ({e}); falling back to interactive choice."
        )

    raw = input("Update policy store? (yes/no): ").strip().lower()
    while raw not in ("yes", "y", "no", "n"):
        print("Invalid answer. Please enter yes or no.")
        raw = input("Update policy store? (yes/no): ").strip().lower()

    if raw in ("yes", "y"):
        return "policy_update"

    return "skip_policy_update"
