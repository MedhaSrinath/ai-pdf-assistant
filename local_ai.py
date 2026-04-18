import requests

url = "http://localhost:11434/api/generate"

user_input = input("Ask something: ")

response = requests.post(
    url,
    json={
        "model": "llama3",
        "prompt": user_input,
        "stream": False,
    },
)

data = response.json()

print("\nAI says:\n")
print(data["response"])