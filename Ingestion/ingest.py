"""Load PDF from files, clean up, split, ingest into a vectorstore"""
import os, sys, logging
import time
from dotenv import load_dotenv
from termcolor import colored

from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import (
    clean_extra_whitespace,
    replace_mime_encodings,
    replace_unicode_quotes,
)


from rich.logging import RichHandler
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.live import Live
from rich.segment import Segment


class ConsolePanel(Console):
    """
    A rich renderable console.
    Applies color to string starting with 1, 2 or 3.
    """

    def __init__(self, *args, **kwargs):
        console_file = open(os.devnull, "w")
        super().__init__(record=True, file=console_file, *args, **kwargs)

    def __rich_console__(self, console, options):
        texts = self.export_text(clear=False).split("\n")
        for line in texts[-options.height :]:
            if len(line) != 0:
                if line[0] == "1":
                    yield f"[green]{line[2:]}"
                elif line[0] == "2":
                    yield f"[yellow]{line[2:]}"
                elif line[0] == "3":
                    yield f"[red]{line[2:]}"
                else:
                    yield line


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="progress", ratio=1),
        Layout(name="main", ratio=1),
    )
    layout["main"].split_row(
        Layout(name="folder", ratio=3),
        Layout(name="body", ratio=7),
    )

    return layout


def ingest_docs(endpoint: str) -> None:
    load_dotenv()
    begin = time.time()
    for folder in range(0, 100):
        folder_start = time.time()

        for file in os.listdir(f"ASN/lettres_de_suivi/{folder}"):
            start_time = time.time()
            job = job_progress.add_task(f"[green]{file}", total=1)
            attempts = 0
            success = False

            while attempts < 4 and not success:
                try:
                    console.print(" ")
                    console.print(f"File: {file}, Attempt n° {attempts}."),

                    loader = UnstructuredFileLoader(
                        f"ASN/lettres_de_suivi/{folder}/{file}",
                        # mode="elements",
                        post_processors=[
                            clean_extra_whitespace,
                            replace_unicode_quotes,
                            replace_mime_encodings,
                        ],
                        # strategy="ocr_only",
                    )
                    raw_documents = loader.load()

                    unrecognized_char = False
                    for doc in raw_documents:
                        for char in doc.page_content:
                            if char == "\uFFFD":
                                unrecognized_char = True
                                break

                    if unrecognized_char:
                        console.print(f"2 Unrecognized characters, switching to OCR.")
                        loader = UnstructuredFileLoader(
                            f"ASN/lettres_de_suivi/{folder}/{file}",
                            # mode="elements",
                            post_processors=[
                                clean_extra_whitespace,
                                replace_unicode_quotes,
                                replace_mime_encodings,
                            ],
                            strategy="ocr_only",
                        )
                        raw_documents = loader.load()

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                    )
                    documents = text_splitter.split_documents(raw_documents)

                    embeddings_model = HuggingFaceInferenceAPIEmbeddings(
                        api_key=os.environ["HF_TOKEN"],
                        api_url=f"{endpoint}",
                        model_name="WhereIsAI/UAE-Large-V1",
                    )

                    for i in range(0, len(documents), 32):
                        vectorstore_db = Chroma.from_documents(
                            documents,
                            embeddings_model,
                            persist_directory="./chroma_db",
                            collection_name="ASN",
                        )

                    console.print(
                        "1 Success! "
                        + f"File {file} done in {round(time.time() - start_time,2)} s!"
                    )
                    job_progress.update(job, completed=1)
                    if len(job_progress.tasks) >= 8:
                        job_progress.remove_task(job_progress.tasks[-8].id)
                    success = True

                except Exception as exception:
                    console.print(f"3 Failure n° {attempts}, {exception}")
                    attempts += 1
                    if attempts == 4:
                        with open("failures_list.txt", "a") as f:
                            f.write(f"{file}\n")
                        break

        for task in job_progress.tasks:
            job_progress.remove_task(task.id)
        console.print(" ---- ")
        console2.print(
            f"1 Folder {folder} done in {round((time.time() - folder_start)/60, 2)}min"
        )
        overall_progress.advance(overall_progress.tasks[0].id)

    console2.print(
        "1 There are", vectorstore_db._collection.count(), "in the collection"
    )
    console2.print(f"1 Total time is {round((time.time() - begin) / 60, 2)}min")
    console2.print(
        f"1 The approximate cost is {round((time.time() - begin) / 60 / 60 * 0.6, 2)}$"
    )
    overall_progress.update(overall_task, completed=100)


if __name__ == "__main__":
    endpoint = input("Enter endpoint adress:")

    ## Rich layout
    console = ConsolePanel()
    console2 = ConsolePanel()

    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    )
    overall_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )
    overall_task = overall_progress.add_task("All Folders", total=100)
    progress_table = Layout()
    progress_table.split_row(
        Panel(
            overall_progress,
            title="Overall Progress",
            border_style="green",
            padding=(2, 2),
        ),
        Panel(
            job_progress,
            title="[b]Jobs",
            border_style="red",
            padding=(2, 2),
        ),
    )
    layout = make_layout()
    layout["progress"].update(progress_table)
    layout["body"].update(Panel(console, title="Output"))

    layout["folder"].update(
        Panel(
            console2,
            title="Infos",
            border_style="yellow",
        )
    )

    # display
    with Live(layout, refresh_per_second=10, screen=True):
        while ingest_docs(endpoint):
            layout["progress"].update(progress_table)
        input("Type a key to quit")
