from pypdf import PdfReader
import io
import os
import re


def clean_data(data):
    # Dehyphenation
    data = re.sub(r'-\s*\n\s*', '', data)

    # Change multiple whitespace to one
    data = re.sub(r'\s+', ' ', data)

    # Remove useless whitespace at the end/beginning
    data = data.strip()

    return data

# Iterate through the raw_data_articles repository
directory = "/Users/martonlori/Documents/GitHub/RAG-Chatbot/data/raw_data_articles"
for entry in os.scandir(directory):

    # For every object there, read it with PDF reader
    reader = PdfReader(entry.path)

    # Open (create) a new file
    new_txt_filename = entry.name.replace(".pdf", ".txt")
    new_txt_path = f"data/plain_text_articles/{new_txt_filename}"
    with open(new_txt_path, mode="a", encoding="utf-8") as writer:

    # Iterate through the pages of the read file, write it to the new file
        pdf_length = reader.get_num_pages()
        for page_number in range(pdf_length):
            dirty_context = reader.pages[page_number].extract_text(extraction_mode="plain")
            context = clean_data(dirty_context)
            writer.write(context)




