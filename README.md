# MCP Server -  Suggest Similarity Diff

This repository contains an MCP server designed to analyze Git differences (diffs) and suggest relevant text modifications based on the cosine similarity to pre-vectorized Markdown files.

## Tools

### `vectorize_md`

This tool processes Markdown files, computes their embeddings using a specified model, and saves the vectors to a JSONL file for later use.

**Note:** This tool is automatically executed when you run the container with `docker run`.

### `diff_similarity`

This tool processes Git diffs, computes embeddings for the changes, and compares them against pre-computed vectors to find similar text modifications.

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/chick-p/suggest-similarity-diff.git
   ```

1. Build the Docker image:

   ```bash
   docker build -t mcp/suggest-similarity-diff:latest .
   ```

## Usage with GitHub Copilot Agent

```json
{
  "mcpServers": {
    "suggest-similarity-diff": {
      "command": "docker",
      "args": [
        "run",
        "-i",
        "--rm",
        "-v",
        "/path/to/my-blog:/app/repo",
        "mcp/suggest-similarity-diff"
      ]
    }
  }
}
```

## Environment Variables

You can pass the following environment variables to the Docker container using the `-e` option:

| Variable                     | Default Value                                 | Description                                                                 |
|------------------------------|-----------------------------------------------|-----------------------------------------------------------------------------|
| `EMBED_MODEL_NAME`           | `paraphrase-multilingual-MiniLM-L12-v2`       | Embedding model name for vectorization.                                      |
| `SIMILARITY_THRESHOLD`       | `0.75`                                       | Cosine similarity threshold for suggestions (range: 0 to 1).                |
| `GIT_DEFAULT_BRANCH_NAME`    | `main`                                       | Default branch name for git diff base.                                       |
| `SETUP_VECTORIZE`            | `false`                                       | Run vectorization (vectorize_md) automatically on container startup.         |
| `INFO_LEVEL`                 | `INFO`                                       | Logging level (`DEBUG`, `INFO`, etc).                                        |
| `ALLOW_SELF_MATCH`           | `false`                                      | Allow self-file matching in similarity search.                               |
| `USE_FAISS`                  | `false`                                      | Use FAISS for fast similarity search (if available).                         |
