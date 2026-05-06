# PostgreSQL & pgvector setup

The agent uses Postgres for:

| Feature | Table / extension |
|--------|-------------------|
| Transcript analysis results | `transcript_analyses` |
| Policy RAG (embeddings) | `policy_chunks` + **`vector`** (pgvector) |

Follow the steps in order. **pgvector is required** if you answer **yes** to “Update policy store?” when running the agent.

---

## 1. Install and start PostgreSQL (macOS / Homebrew)

```bash
brew install postgresql@16
brew services start postgresql@16
brew services list   # postgresql@16 should show "started"
```

---

## 2. Create the database

```bash
createdb customer_support
```

Use your Mac username as the DB user (typical for Homebrew), or create a dedicated user (see below).

---

## 3. Set user and password in `.env`

The app reads `POSTGRES_*` from `.env` (or `DATABASE_URL`).

**Typical Homebrew setup (user = your Mac login):**

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=customer_support
POSTGRES_USER=your_username    # output of: whoami
POSTGRES_PASSWORD=postgres     # set after ALTER USER (see below)
```

Set a password so tools like DataGrip work:

```bash
psql -d customer_support -c "ALTER USER your_username PASSWORD 'postgres';"
```

**Or use one URL:**

```env
DATABASE_URL=postgresql://your_username:postgres@localhost:5432/customer_support
```

---

## 4. Install pgvector (required for policy ingest)

Without this, creating `policy_chunks` fails with: `type "vector" does not exist`.

### 4a. Try Homebrew first

```bash
brew install pgvector
```

Check that the extension file exists **for your Postgres version** (change `@16` if you use 14/15/17):

```bash
ls "$(brew --prefix postgresql@16)/share/postgresql@16/extension/vector.control"
```

If that file **exists**, enable the extension:

```bash
psql -d customer_support -c "CREATE EXTENSION vector;"
```

### 4b. Error: `extension "vector" is not available` / `vector.control: No such file`

Homebrew’s `pgvector` bottle is often built for **default** `postgresql`, not `postgresql@16`. Your server is **postgresql@16**, so the extension files never land in the `@16` folder.

**Fix — build pgvector against Postgres 16** (Apple Silicon: `/opt/homebrew`; Intel Mac: often `/usr/local`):

```bash
# Path to YOUR running Postgres (must match brew services)
export PG_CONFIG="$(brew --prefix postgresql@16)/bin/pg_config"

cd /tmp
rm -rf pgvector
git clone --branch v0.8.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

Then:

```bash
psql -d customer_support -c "CREATE EXTENSION vector;"
```

If `make install` says permission denied, run the same `make install` with a user that can write into `$(brew --prefix postgresql@16)` (on Homebrew installs that is usually your user, no sudo).

### 4c. Alternative: Postgres + pgvector in Docker

If you prefer not to compile:

```bash
docker run -d --name pgvector -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=customer_support -p 5432:5432 pgvector/pgvector:pg16
```

Then in `.env`:

```env
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=customer_support
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
```

Stop any local `brew services` Postgres on 5432 first, or map Docker to another port (e.g. `-p 5433:5432` and set `POSTGRES_PORT=5433`).

### Permission denied on `CREATE EXTENSION`

Connect as a superuser and run against `customer_support`:

```bash
psql -d customer_support -U postgres -c "CREATE EXTENSION vector;"
```

---

## 5. Python dependencies

From the project root (use the project venv):

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 6. Verify the database

```bash
python view_db.py
```

- **Connected:** you should see `Found N row(s)` (often `0` before running the agent).
- **Connection failed:** start Postgres, fix `POSTGRES_*` / password, then retry.

---

## 7. Optional: GUI (DataGrip / DBeaver)

| Field | Example |
|-------|---------|
| Host | `localhost` |
| Port | `5432` |
| Database | `customer_support` |
| User | same as `POSTGRES_USER` in `.env` |
| Password | same as `POSTGRES_PASSWORD` |

---

## 8. What gets created automatically

On first use, the app creates (if missing):

- **`transcript_analyses`** — LLM transcript summaries (no extension needed).
- **`policy_chunks`** — chunked policy text + embeddings (**needs pgvector**).

---

## 9. Embedding model

Default is **1536** dimensions (`text-embedding-3-small`). If you change `OPENAI_EMBEDDING_MODEL` in `.env`, update **`POLICY_EMBEDDING_DIMENSIONS`** in `db.py` and recreate or migrate `policy_chunks`.

---

## 10. Policy files

Put **`.txt`** policy files under `assets/policies/`. Ingest runs when you run the agent and choose **yes** to update the policy store. See `assets/policies/README.md`.
