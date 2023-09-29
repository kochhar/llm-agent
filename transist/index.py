import collections
import glob
import json
import logging
import random
from os import path
from pathlib import Path

import chromadb
import numpy as np
from chromadb import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from langchain.document_loaders import TextLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, utils
from typing import Dict, List, Sequence, Type


log = logging.getLogger(__name__)


def canonical_section(section: str) -> str:
    return section.lower().replace(' ', '_')


class PddCollection(object):
    """Represents a collection of PDD documents indexed by sections.

    Supports searching the collection constrained by section as well as other
    metadata constraints based on document metadata.
    """
    def __init__(self, index_dir, sections: List[str]):
        canonical_sections = map(canonical_section, sections)
        settings = Settings(anonymized_telemetry=False)
        self.pdd_indexes: Dict[str, SectionIndex] = {
            section: SectionIndex(section, chromadb.PersistentClient(path.join(index_dir, section), settings))
            for section in canonical_sections
        }

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
        files_by_section = collections.defaultdict(list)
        metadata_by_section = collections.defaultdict(list)

        for pdd_dir in pdd_dirs:
            log.info("Indexing PDD from directory: %s", pdd_dir)
            for text_file in glob.glob(f"{pdd_dir}/sections/*.txt"):
                section = self._section_from_file_name(path.basename(text_file))
                files_by_section[section].append(text_file)

            for metadata_file in glob.glob(f"{pdd_dir}/sections/*.json"):
                section = self._section_from_file_name(path.basename(metadata_file))
                metadata_by_section[section].append(metadata_file)

        for section in files_by_section:
            try:
                self.pdd_indexes[section].index_filepaths(filepaths=files_by_section[section],
                                                          metadatapaths=metadata_by_section[section])
            except KeyError:
                log.warning("Unknown section %s, skipping %d pdd files",
                            section, len(files_by_section[section]))

    def search_in_section(self, section, query, **kwargs) -> Sequence[Document]:
        return self.pdd_indexes[canonical_section(section)].search(query, **kwargs)

    @staticmethod
    def _section_from_file_name(file_name: str) -> str:
        base_file_name, section_str = file_name.rsplit('_', 1)
        section_name: str = section_str.split(' ', 1)[1].split('.', 1)[0]
        return canonical_section(section_name)


class SectionIndex(object):
    def __init__(self, section_name, index_client: chromadb.API):
        self.section_name = section_name
        self._pdd_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
        )
        self.client = index_client
        self.collection = self.client.get_or_create_collection(
            name=f"chromadb-{self.section_name}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=SentenceTransformerEmbeddingFunction(model_name="multi-qa-mpnet-base-dot-v1")
        )

    def index_filepaths(self, filepaths: List[str], metadatapaths: List[str]):
        for filepath, metadatapath in zip(filepaths, metadatapaths):
            metadata = json.loads(Path(metadatapath).read_text())
            loader = PddSectionLoader(self.section_name, filepath, metadata)
            chunks = self._pdd_splitter.split_documents(loader.load())
            if random.random() < 0.2:
                for i in range(10):
                    log.debug("Chunk: %s", random.choice(chunks))

            ids = [f"{filepath}:{i}" for i in range(len(chunks))]
            metadatas = [c.metadata for c in chunks]
            texts = [c.page_content for c in chunks]
            self.collection.upsert(ids=ids,
                                   metadatas=metadatas,
                                   documents=texts)
            log.info("Upserted %d chunks from: '%s'", len(chunks), filepath)

    def search(self, query: str, where_meta=None, n: int = 10, **kwargs) -> Sequence[Document]:
        log.info("Searching %s collection", self.section_name)
        results = self.collection.query(
            query_texts=[query],
            n_results=int(n*1.5),
            where=where_meta,
            include=["metadatas", "documents", "embeddings"],
            **kwargs
        )

        candidates = [
            Document(page_content=d, metadata=m or {})
            for (d, m) in zip(results["documents"][0], results["metadatas"][0])
        ]
        log.info("Got %i candidate items", len(candidates))

        query_embeddings = self.collection._embedding_function([query])
        mmr_selected = utils.maximal_marginal_relevance(
            np.array(query_embeddings[0], np.float32),
            results["embeddings"][0],
            k=n,
            lambda_mult=0.5
        )
        log.info("Selected %i most diverse results", len(mmr_selected))

        return [doc for (i, doc) in enumerate(candidates) if i in mmr_selected]


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


