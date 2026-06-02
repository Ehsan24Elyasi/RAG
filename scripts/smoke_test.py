import requests

print("Health:", requests.get("http://127.0.0.1:8000/health", timeout=30).text)
print("Ingest:", requests.post("http://127.0.0.1:8000/ingest", timeout=120).text)
print(
    "Query:",
    requests.post(
        "http://127.0.0.1:8000/query",
        json={"question": "Basalam چیه؟", "debug": True},
        timeout=120,
    ).text,
)
