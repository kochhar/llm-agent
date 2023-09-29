import argparse
import glob
import logging

from transist import index


SECTIONS = [
    "project details",
    "safeguards",
    "application of methodology",
    "quantification of ghg emission",
    "monitoring"
]


def search_index(index_dir):
    pdd_coll = index.PddCollection(index_dir, sections=SECTIONS)
    results = pdd_coll.search_in_section("application of methodology", "stove", n=5)
    for result in results:
        print(f"""text: {result.page_content}\n\nmetadata: {result.metadata}\n\n\n""")


def make_index(pdd_dir, index_dir):
    index.PddCollection.from_directories(glob.glob(f"{pdd_dir}/*"), index_dir,
                                         sections=SECTIONS)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        prog="rag_index",
        description="Manipulate Transistry RAG index"
    )
    parser.add_argument("-i", "--index_dir", action="store", required=True)
    parser.add_argument("-d", "--data_dir", action="store", default=None, required=False)

    args = parser.parse_args()
    if args.data_dir:
        make_index(pdd_dir=args.data_dir, index_dir=args.index_dir)
    search_index(args.index_dir)
