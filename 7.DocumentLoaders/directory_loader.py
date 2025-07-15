from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path = '7.DocumentLoaders\\Books',
    glob = '*.pdf',
    loader_cls = PyPDFLoader
)

docs = loader.lazy_load()

count = 0
for documents in docs:
    count = count + 1

print(count)