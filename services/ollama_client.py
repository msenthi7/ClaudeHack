import ollama


def run_ollama_llm(prompt: str, model: str = "llama3.1"):
    try:
        response = ollama.generate(model="llama3.1", prompt=prompt)
        # Check what attributes the response has, for example:
        # print(dir(response))
        # If it contains 'completion', use that:
        return response["response"]
        # Alternatively, sometimes it is response.text or response.choices[0].text based on the client
    except Exception as e:
        return f"ERROR: {e}"
