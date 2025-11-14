import importlib
mods = [
 "langchain_community.document_loaders",
 "langchain_text_splitters",
 "langchain_huggingface",
 "langchain_community.vectorstores",
 "langchain_ollama",
 "langchain.chains"
]
for m in mods:
    try:
        importlib.import_module(m)
        print("OK:", m)
    except Exception as e:
        print("FAIL:", m, "->", type(e).__name__, e)
