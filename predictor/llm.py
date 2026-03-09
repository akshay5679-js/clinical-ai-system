import ollama

def ask_llm(question):

    response = ollama.chat(
        model='llama3',
        messages=[
            {
                'role': 'user',
                'content': question,
            },
        ],
    )

    return response['message']['content']