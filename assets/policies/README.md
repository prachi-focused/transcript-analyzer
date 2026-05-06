# Policy documents (TXT)

Place **`.txt`** files here. The policy ingest pipeline will:

1. Load each file  
2. Split on markdown-style headings (`#`, `##`, …) into sections; text before the first heading is **(preamble)**; files with no headings are one section **(whole document)**  
3. Chunk and embed with OpenAI  
4. Store rows in Postgres **pgvector** with metadata for citations  

**Metadata per chunk:** `source=policy_txt`, `document_name`, `section_heading`, `chunk_index`, `uploaded_at`, plus `content`.

### Headings

Use lines like `# Title` or `## Section` so sections are labeled correctly in Q&A citations.

### Run ingest

```bash
python -c "from node_2_policy_update import run_policy_ingest_pipeline; print(run_policy_ingest_pipeline())"
```

Or run the agent and choose **yes** to update the policy store.

(Requires venv, `OPENAI_API_KEY`, Postgres + pgvector — see `SETUP_DB.md`.)
