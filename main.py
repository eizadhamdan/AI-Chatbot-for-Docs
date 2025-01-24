import tkinter as tk
from tkinter import scrolledtext, messagebox
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Function to handle query processing
def process_query():
    query_text = user_input.get("1.0", tk.END).strip()
    if not query_text:
        messagebox.showwarning("Input Error", "Please enter a query.")
        return

    try:
        # Prepare the DB
        embedding_function = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

        # Search the DB
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        if len(results) == 0 or results[0][1] < 0.7:
            response_output.config(state=tk.NORMAL)
            response_output.delete("1.0", tk.END)
            response_output.insert(tk.END, "Unable to find matching results.")
            response_output.config(state=tk.DISABLED)
            return

        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)

        model = ChatOpenAI()
        response_text = model.predict(prompt)

        sources = [doc.metadata.get("source", None) for doc, _score in results]
        formatted_response = f"Response: {response_text}\n\nSources: {', '.join(filter(None, sources))}"

        response_output.config(state=tk.NORMAL)
        response_output.delete("1.0", tk.END)
        response_output.insert(tk.END, formatted_response)
        response_output.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# Create the main Tkinter window
root = tk.Tk()
root.title("Alice in Wonderland Chatbot")
root.geometry("800x600")
root.configure(bg="#F3E5F5")  # Light lavender background

# Add a title label
title_label = tk.Label(
    root,
    text="Welcome to Alice in Wonderland Chatbot!",
    font=("Roboto", 16, "bold"),
    bg="#9575CD",  # Purple color
    fg="white",
    pady=10,
)
title_label.pack(fill=tk.X)

# Input area
input_frame = tk.Frame(root, bg="#D1C4E9", padx=10, pady=10)  # Light purple frame
input_frame.pack(pady=10, padx=10, fill=tk.BOTH)

tk.Label(
    input_frame,
    text="Enter your query below:",
    font=("Roboto", 12),
    bg="#D1C4E9",
    fg="black",
).pack(anchor="w", pady=5)

user_input = scrolledtext.ScrolledText(
    input_frame, wrap=tk.WORD, height=5, width=60, font=("Roboto", 11)
)
user_input.pack(pady=5)

# Submit button
submit_button = tk.Button(
    root,
    text="Submit",
    command=process_query,
    bg="#673AB7",  # Deep purple
    fg="white",
    font=("Roboto", 12, "bold"),
    relief=tk.RAISED,
    padx=10,
    pady=5,
)
submit_button.pack(pady=10)

# Output area
output_frame = tk.Frame(root, bg="#EDE7F6", padx=10, pady=10)  # Very light purple frame
output_frame.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

tk.Label(
    output_frame,
    text="Response from Chatbot:",
    font=("Roboto", 12),
    bg="#EDE7F6",
    fg="black",
).pack(anchor="w", pady=5)

response_output = scrolledtext.ScrolledText(
    output_frame, wrap=tk.WORD, height=12, width=60, font=("Roboto", 11), bg="#FFF8E1"
)
response_output.pack(pady=5)
response_output.config(state=tk.DISABLED)

# Footer
footer_label = tk.Label(
    root,
    text="Made by Eizad Hamdan",
    font=("Helvetica", 16, "italic"),
    bg="#F3E5F5",
    fg="#512DA8",  # Dark purple
    pady=10,
)
footer_label.pack(side=tk.BOTTOM, fill=tk.X)

# Run the GUI event loop
root.mainloop()
