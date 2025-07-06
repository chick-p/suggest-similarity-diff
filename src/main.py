import logging
import os
from .mcp_app import mcp

from .tools import diff_similarity
from .tools import vectorize_md

if __name__ == "__main__":

    run_vectorize = os.environ.get("SETUP_VECTORIZE", "false").lower()
    if run_vectorize == "true":
        vectorize_md.vectorize_md()

    mcp.run()
