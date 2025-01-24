import tkinter as tk
from tkinter import messagebox
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
        # Prepare the DB.
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB.
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

        # Clean up the response to remove metadata
        response_text = str(response).split("additional_kwargs")[0].strip()

        sources = [doc.metadata.get("id", None) for doc, _score in results]
        formatted_response = f"Response:\n{response_text}\n\nSources:\n{', '.join(filter(None, sources))}"
        return formatted_response
    except Exception as e:
        return f"An error occurred: {str(e)}"


def on_submit():
    query_text = query_entry.get("1.0", tk.END).strip()
    if not query_text:
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    # Clear the output area before displaying the new response
    output_text.delete("1.0", tk.END)

    response = query_rag(query_text)
    output_text.insert(tk.END, response)

# Create the GUI
root = tk.Tk()
root.title("AI for PDFs")
root.geometry("800x600")
root.configure(bg="#8B5CF6")

# Input label and text area
input_label = tk.Label(root, text="Enter your query:")
input_label.pack(pady=5)

query_entry = tk.Text(root, height=5, width=50)
query_entry.pack(pady=5)

# Submit button
submit_button = tk.Button(
    root,
    text="Submit",
    command=on_submit,
    bg="#673AB7",
    fg="white",
    font=("Roboto", 12, "bold"),
    relief=tk.RAISED,
    padx=10,
    pady=5,
)
submit_button.pack(pady=10)

# Output label and text area
output_label = tk.Label(root, text="Response:")
output_label.pack(pady=5)

output_text = tk.Text(root, height=15, width=50, state="normal")
output_text.pack(pady=5)

footer_label = tk.Label(
    root,
    text="Made by Eizad Hamdan",
    font=("Helvetica", 16, "italic"),
    bg="#F3E5F5",
    fg="#512DA8",  # Dark purple
    pady=10,
)
footer_label.pack(side=tk.BOTTOM, fill=tk.X)

# Run the GUI loop
root.mainloop()
