"""Minimal WSGI app to avoid deployment 404s on platforms expecting `app` callable.

This repository's interactive UI is Streamlit-based (`app/interface.py`) for local use.
"""


def app(environ, start_response):
    path = environ.get("PATH_INFO", "/")

    if path in ("/", ""):
        body = (
            "trading-bot is running.\\n"
            "For local interactive UI run: streamlit run app/interface.py\\n"
        ).encode("utf-8")
        start_response("200 OK", [("Content-Type", "text/plain; charset=utf-8")])
        return [body]

    if path == "/health":
        start_response("200 OK", [("Content-Type", "text/plain; charset=utf-8")])
        return [b"ok"]

    start_response("404 Not Found", [("Content-Type", "text/plain; charset=utf-8")])
    return [b"not found"]
