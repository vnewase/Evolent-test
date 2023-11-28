#ENter api key in below variable before executing script
OPENAI_API_KEY = ""

import tkinter as tk
from tkinter import scrolledtext
from tkinter import ttk
from IPython.display import display, clear_output

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Annoy
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import sys
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    memory=memory
)
embeddings_func = HuggingFaceEmbeddings()
vector_store_load = Annoy.load_local(
    "./Data/vector_embeddings_classification_answers", embeddings=embeddings_func
)


def get_class(context):
    context_text = []
    for i in context:
        if i[1] < 1:
            txt = str(i[0]).replace("page_content=", '')
            context_text.append(txt)
    if context_text:
        classification_title = context_text[0].split("\\n")[0].split(":")[1].strip()
        context_out = "\n".join(context_text)
    else:
        classification_title = 'Not able to classify'
        context_out = "No relevant data or less data, could not classify and find similar documents"

    return classification_title, context_out


# class_title_display= "Not classified"
def classify_text(entry, text_widget1, output_label1):
    text = entry.get()
    similar_text = vector_store_load.similarity_search_with_score(text, k=3)
    class_title, results = get_class(similar_text)
    print(class_title)
    text_widget1.delete(1.0, tk.END)  # Clear previous content
    text_widget1.insert(tk.END, results)
    output_label1.config(text=f"{class_title}")

    # label1 = tk.Label(window, text=class_title)
    # label2 = tk.Label(window, text="Processed Output:")
    # label1.grid(row=3, column=0, columnspan=2, pady=10)


def answer_query(entry, text_widget2):
    text = entry.get()
    result = get_answer(text)
    text_widget2.delete(1.0, tk.END)  # Clear previous content
    text_widget2.insert(tk.END, result['result'])


def close_ui(window):
    window.destroy()
    sys.exit()


def create_gui():
    # Create the main window
    window = tk.Tk()
    window.title("Medical text bot")

    # Create input boxes and labels
    entry1 = tk.Entry(window, width=40)
    entry2 = tk.Entry(window, width=40)
    entry1.grid(row=0, column=0, padx=10, pady=10, sticky="w")
    entry2.grid(row=0, column=2, padx=10, pady=10, sticky="w")

    label_var1 = tk.StringVar()
    label_var2 = tk.StringVar()
    label1 = tk.Label(window, textvariable=label_var1)
    label2 = tk.Label(window, textvariable=label_var2)
    label1.grid(row=1, column=0, padx=10, pady=10, sticky="w")
    label2.grid(row=2, column=0, padx=10, pady=10, sticky="w")

    # Create submit buttons
    button1 = ttk.Button(window, text="Classify", command=lambda: classify_text(entry1, text_widget1, output_label1))
    button2 = ttk.Button(window, text="Search query", command=lambda: answer_query(entry2, text_widget2))
    button1.grid(row=0, column=1, padx=10, pady=10, sticky="w")
    button2.grid(row=0, column=3, padx=10, pady=10, sticky="w")

    # Create processed output labels
    output_label1 = tk.Label(window, text="Class:")
    output_label2 = tk.Label(window, text="Answer:")
    output_label1.grid(row=3, column=0, padx=10, pady=10, columnspan=2, sticky="w")
    output_label2.grid(row=3, column=2, padx=10, pady=10, columnspan=2, sticky="w")

    # Create display boxes with scrollbar
    text_widget1 = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=5, width=40)
    text_widget2 = scrolledtext.ScrolledText(window, wrap=tk.WORD, height=5, width=40)
    text_widget1.grid(row=4, column=0, padx=10, pady=10, columnspan=2, sticky="w")
    text_widget2.grid(row=4, column=2, padx=10, pady=10, columnspan=2, sticky="w")

    # Run the main loop
    close_button = ttk.Button(window, text="Close", command=lambda: close_ui(window))
    close_button.grid(row=5, column=0, columnspan=4, pady=10)
    window.mainloop()


# Set the matplotlib backend to TkAgg for GUI compatibility
# %matplotlib tk
# create_gui()

if __name__ == "__main__":
    create_gui()
