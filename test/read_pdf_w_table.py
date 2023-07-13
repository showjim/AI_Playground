from langchain.document_loaders import (
    PDFMinerLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    TextLoader,
)
import camelot
from ctypes.util import find_library
find_library("gs")

loader = PyMuPDFLoader("./foo.pdf")
documents = loader.load()
# print(documents[0].page_content)

tables = camelot.read_pdf('./foo.pdf')
a = tables[0]
b = a.df
print(b)