"""
Project Recallix Web Application Launcher

Usage:
    python run_server.py
"""

import sys
import uvicorn

if __name__ == "__main__":
    # Ensure stdout handles utf-8 encoding safely on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=" * 65)
    print(" 🧠 Project Recallix — AI Second Brain Web Application")
    print("=" * 65)
    print(" -> Local Web App URL: http://localhost:8000/")
    print(" -> Interactive API Docs (Swagger): http://localhost:8000/docs")
    print(" -> Alternative API Docs (ReDoc):  http://localhost:8000/redoc")
    print("=" * 65)
    print("Press CTRL+C to stop the server.\n")

    uvicorn.run(
        "app.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
