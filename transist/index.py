import collections
import glob
import json
import logging
from os import path
from pathlib import Path

import chromadb
from chromadb import Settings

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import OpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, utils
from llama_index import ServiceContext, SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_hub.file.pdf.base import PDFReader
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.vector_stores import ChromaVectorStore

from typing import Dict, List, Sequence


log = logging.getLogger(__name__)
chunklog = logging.getLogger(f"{__name__}.chunks")


def canonical_section(section: str) -> str:
    return section.lower().replace(' ', '_')


class BaseCollection(object):
    def __init__(self, index_dir, collection_name):
        self.index_dir = index_dir
        self.client = chromadb.PersistentClient(
            path=self.index_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embed_model = "local"
        self.vector_store = ChromaVectorStore(
            chroma_collection=self.client.get_or_create_collection(collection_name)
        )
        self.index = None

    def answer(self, query):
        if not self.index:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                service_context=ServiceContext.from_defaults(embed_model=self.embed_model)
            )
        return self.index.as_query_engine(similarity_top_k=5).query(query)

    def search(self, query, n=5, mmr=False):
        if not self.index:
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                service_context=ServiceContext.from_defaults(embed_model=self.embed_model)
            )
        return self.index.as_retriever(similarity_top_k=n).retrieve(query)


class MethodologyCollection(BaseCollection):
    def __init__(self, index_dir):
        super().__init__(index_dir=index_dir, collection_name="methodology_index")

    @classmethod
    def from_directory(cls, methodology_dir, index_dir):
        instance = cls(index_dir=index_dir)
        instance.index_directory(methodology_dir)
        return instance

    def index_directory(self, methodology_dir):
        loader = SimpleDirectoryReader(input_dir=methodology_dir,
                                       required_exts=[".pdf"],
                                       file_extractor={'.pdf': PDFReader()})

        documents = loader.load_data()
        log.info("Indexing %i documents in index", len(documents))
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=StorageContext.from_defaults(vector_store=self.vector_store),
            service_context=ServiceContext.from_defaults(embed_model=self.embed_model),
            show_progress=True
        )
        log.debug("Done!")


class PddLlamaCollection(BaseCollection):
    def __init__(self, index_dir):
        super().__init__(index_dir=index_dir, collection_name="pdd_index")

    @classmethod
    def from_directory(cls, pdd_dir, index_dir):
        instance = cls(index_dir=index_dir)
        instance.index_directories(pdd_dir)
        return instance

    def index_directories(self, pdd_dir):
        loader = SimpleDirectoryReader(input_dir=pdd_dir,
                                       required_exts=[".txt"],
                                       recursive=True,
                                       file_metadata=pdd_section_file_metadata)
        documents = loader.load_data()
        log.info("Indexing %i documents in index", len(documents))
        self.index = VectorStoreIndex.from_documents(
            documents=documents,
            storage_context=StorageContext.from_defaults(vector_store=self.vector_store),
            service_context=ServiceContext.from_defaults(embed_model=self.embed_model),
            show_progress=True
        )
        log.debug("Done!")


def section_from_file_name(file_name: str) -> str:
    base_file_name, section_str = file_name.rsplit('_', 1)
    section_name: str = section_str.split(' ', 1)[1].split('.', 1)[0]
    return canonical_section(section_name)


def pdd_section_file_metadata(filename: str):
    section = section_from_file_name(path.basename(filename))
    meta_filepath = filename.strip('.txt') + '.json'
    metadata = json.loads(Path(meta_filepath).read_text())
    metadata.update({
        'section_name': section
    })
    return metadata


class PddCollection(object):
    """Represents a collection of PDD documents indexed by sections.

    Supports searching the collection constrained by section as well as other
    metadata constraints based on document metadata.
    """
    def __init__(self, index_dir, sections):
        self.index_dir = index_dir
        self._pdd_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
        )
        self.embedding = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
        self.persistent_client = chromadb.PersistentClient(self.index_dir, Settings(anonymized_telemetry=False))
        self.pdd_index = Chroma(
            collection_name="pdd_index",
            embedding_function=self.embedding,
            persist_directory=self.index_dir,
            client=self.persistent_client,
            collection_metadata={"hnsw:space": "cosine"})

    @classmethod
    def from_directories(cls, pdd_dirs: List[str], index_dir: str, sections: List[str]):
        instance = cls(index_dir, sections)
        instance.index_directories(pdd_dirs)
        return instance

    def index_directories(self, pdd_dirs: List[str]):
        """
        Indexes PDDs divided up into .txt and .json files per section.

        Expects each pdd_dir in `pdd_dirs` to contain a subdirectory /sections/
        which contain pairs of files for each section extracted from the PDD.

            <PDD NAME>_# <SECTION>.txt and
            <PDD_NAME>_# <SECTION<.json

        :param pdd_dirs: List of directories, one per PDD
        """
        files_by_section, metadata_by_section = self.files_by_section(pdd_dirs)
        for section in map(canonical_section, files_by_section):
            log.info("Indexing section: %s", section)

            section_chunks = []
            for filepath, meta_filepath in zip(files_by_section[section], metadata_by_section[section]):
                metadata = json.loads(Path(meta_filepath).read_text())
                loader = PddSectionLoader(section, filepath, metadata)
                doc_chunks = self._pdd_splitter.split_documents(loader.load())
                log.debug("Got %i chunks from %s", len(doc_chunks), filepath)
                section_chunks.extend(doc_chunks)

            log.info("Got %i chunks to index for section %s", len(section_chunks), section)
            self.pdd_index.add_documents(section_chunks)

        self.pdd_index.persist()

    @staticmethod
    def files_by_section(pdd_dirs: List[str]):
        files_by_section = collections.defaultdict(list)
        metadata_by_section = collections.defaultdict(list)

        for pdd_dir in pdd_dirs:
            log.info("Indexing PDD from directory: %s", pdd_dir)
            for text_file in glob.glob(f"{pdd_dir}/sections/*.txt"):
                section = section_from_file_name(path.basename(text_file))
                files_by_section[section].append(text_file)

            for metadata_file in glob.glob(f"{pdd_dir}/sections/*.json"):
                section = section_from_file_name(path.basename(metadata_file))
                metadata_by_section[section].append(metadata_file)
        return files_by_section, metadata_by_section

    def search_in_section(self, section, query, n=5, mmr=True) -> Sequence[Document]:
        db = self.pdd_index
        section_retriever = db.as_retriever(
            search_type="mmr" if mmr else "similarity",
            search_kwargs={
                'filter': {'section_name': canonical_section(section)},
                'k': 5,
                'fetch_k': int(n*2)
            }
        )
        return section_retriever.get_relevant_documents(query)


class PddSectionLoader(TextLoader):
    def __init__(self, section_name, file_path, metadata, *args, **kwargs):
        super().__init__(file_path, *args, **kwargs)
        self.section_name = section_name
        self.metadata = metadata

    def load(self) -> List[Document]:
        docs = super().load()
        docs[0].metadata.update(self.metadata)
        docs[0].metadata.update({
            'filename': path.basename(self.file_path),
            'section_name': self.section_name
        })
        return docs
