import tkinter as tk
from tkinter import messagebox, scrolledtext
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from embedding_function import get_embedding_function
import os
from dotenv import load_dotenv


load_dotenv()
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def query_rag(query_text: str):
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        results = db.similarity_search_with_score(query_text, k=5)
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        api_key = os.getenv("OPENAI_API_KEY")
        model = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            openai_api_key=api_key
        )
        response = model.invoke(prompt)

        response_text = response.content.strip()
        sources = [
            f"{doc.metadata.get('id', 'unknown source')} (page {doc.metadata.get('page', 'unknown')})"
            for doc, _score in results
        ]
        sources_text = "\n".join(sources)

        formatted_response = (
            f"Response:\n{response_text}\n\n"
            f"Sources:\n{sources_text}"
        )

        return formatted_response
    except Exception as e:
        return f"An error occurred: {str(e)}"


def create_input_area(root):
    frame = tk.Frame(root, bg="#F7FEE7", pady=10)
    frame.pack(fill=tk.X)

    label = tk.Label(frame, text="Enter your query:", font=("Roboto", 12), bg="#E4E4E7", fg="#09090B")
    label.pack(anchor="w", padx=10)

    text_area = scrolledtext.ScrolledText(frame, height=5, wrap=tk.WORD, font=("Roboto", 12))
    text_area.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

    return text_area


def create_output_area(root):
    frame = tk.Frame(root, bg="#F7FEE7", pady=10)
    frame.pack(fill=tk.BOTH, expand=True)

    label = tk.Label(frame, text="Response:", font=("Roboto", 12), bg="#E4E4E7", fg="#09090B")
    label.pack(anchor="w", padx=10)

    text_area = scrolledtext.ScrolledText(frame, height=15, wrap=tk.WORD, font=("Roboto", 12), state="normal")
    text_area.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)

    return text_area


def on_submit(query_entry, output_text):
    query_text = query_entry.get("1.0", tk.END).strip()
    if not query_text:
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    output_text.delete("1.0", tk.END)
    response = query_rag(query_text)
    output_text.insert(tk.END, response)


def on_clear(query_entry, output_text):
    query_entry.delete("1.0", tk.END)
    output_text.delete("1.0", tk.END)


def create_buttons(root, query_entry, output_text):
    frame = tk.Frame(root, bg="#2C3E50")
    frame.pack(pady=10)

    submit_button = tk.Button(frame, text="Submit", font=("Roboto", 12, "bold"), bg="#CC0000", fg="white",
                              command=lambda: on_submit(query_entry, output_text))
    submit_button.pack(side=tk.LEFT, padx=10)

    clear_button = tk.Button(frame, text="Clear", font=("Roboto", 12, "bold"), bg="#FFC72C", fg="white",
                             command=lambda: on_clear(query_entry, output_text))
    clear_button.pack(side=tk.LEFT, padx=10)


def create_footer(root):
    footer = tk.Label(
        root,
        text="Made by Eizad Hamdan",
        font=("Helvetica", 16, "italic"),
        bg="#0F52BA",
        fg="#98FB98",
        pady=10,
    )
    footer.pack(side=tk.BOTTOM, fill=tk.X)


def main():
    root = tk.Tk()
    root.title("AI for PDFs")
    root.geometry("800x600")
    root.configure(bg="#0F52BA")

    query_entry = create_input_area(root)
    output_text = create_output_area(root)
    create_buttons(root, query_entry, output_text)
    create_footer(root)

    root.mainloop()


if __name__ == "__main__":
    main()
