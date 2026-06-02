import requests

resp = requests.post("http://127.0.0.1:8000/ingest", timeout=120)
print(resp.status_code)
print(resp.text)
