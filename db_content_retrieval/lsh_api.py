import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import json

class ColumnContentSearcher:
    def __init__(self, source_db_content_index_paths, synonym_sources):
        """
        Initialize the ColumnContentSearcher with multiple sources.

        Parameters:
            source_db_content_index_paths (dict): A dictionary mapping source names to their db_content_index_paths.
        """
        self.source_db_content_index_paths = source_db_content_index_paths
        self.searcher = {}

        for source, db_content_index_path in source_db_content_index_paths.items():
            db_ids = os.listdir(db_content_index_path)
            self.searcher[source] = {}
            for db_id in tqdm(db_ids, desc=f"Loading searchers for source '{source}'"):
                table_column_indexes = os.listdir(os.path.join(db_content_index_path, db_id))
                for table_column_index in table_column_indexes:
                    try:
                        table, column = table_column_index.split("-**-")
                        # Initialize the searcher for each column
                        searcher_path = os.path.join(db_content_index_path, db_id, table_column_index)
                        # check if there's a subdirectory and only 1 subdirectory in the searcher_path
                        if os.path.isdir(searcher_path):
                            subdirs = os.listdir(searcher_path)
                            if len(subdirs) == 1:
                                searcher_path = os.path.join(searcher_path, subdirs[0])
                        
                                column = column + "/" + subdirs[0]

                        # if not args.lazy_load:
                        lucene_searcher = LuceneSearcher(searcher_path)
                        lucene_searcher.set_bm25(k1=1.2, b=0.75)
                        self.searcher[source].setdefault(db_id, {}).setdefault(table, {})[column] = lucene_searcher
                        # else:
                        #     self.searcher[source].setdefault(db_id, {}).setdefault(table, {})[column] = searcher_path
                    except Exception as e:
                        print(f"Error loading {source}/{db_id}/{table_column_index}: {e}")

        for source in synonym_sources:
            self.searcher[source] = self.searcher[synonym_sources[source]]
        
    def get_searcher(self, source, db_id, table, column):
        """
        Retrieve the searcher for the given source, db_id, table, and column.
        """
        searcher = self.searcher.get(source, {}).get(db_id, {}).get(table, {}).get(column, None)
        # if type(searcher) == str:
        #     searcher = LuceneSearcher(searcher)
        #     searcher.set_bm25(k1=1.2, b=0.75)
        return searcher

    def search_column_content(self, source, db_id, table, column, query, k=10):
        """
        Search the column content for a given query.

        Parameters:
            source (str): The source name (e.g., 'bird-dev', 'bird-train').
            db_id (str): The database identifier.
            table (str): The table name.
            column (str): The column name.
            query (str): The search query.
            k (int): The number of results to return.

        Returns:
            list: A list of search results, or None if the searcher is not found.
        """
        searcher = self.get_searcher(source, db_id, table, column)
        if searcher is None:
            return None
        hits = searcher.search(query, k=k)
        if len(hits) > 0:
            results = [json.loads(hit.raw) for hit in hits]
            results = [x['contents'] for x in results]
        elif searcher.num_docs > 0:
            results = [json.loads(searcher.doc(0).raw())['contents']]
        else:
            results = []

        # if args.lazy_load:
        #     del searcher
        #     gc.collect()
        return results

# FastAPI app
app = FastAPI(title="Column Content Searcher API")

# Initialize the ColumnContentSearcher in the startup event
column_content_searcher = None

@app.on_event("startup")
def startup_event():
    global column_content_searcher
# Replace 'your_db_content_index_path' with the actual path to your index
    test_set_names = [
        # DB Schema related
        'DB_schema_synonym',
        'DB_schema_abbreviation',
        'DB_DBcontent_equivalence',

        # NLQ related
        'NLQ_keyword_synonym',
        'NLQ_keyword_carrier',
        'NLQ_column_synonym',
        'NLQ_column_carrier',
        'NLQ_column_attribute',
        'NLQ_column_value',
        'NLQ_value_synonym',
        'NLQ_multitype',
        'NLQ_others',
        # SQL related
        'SQL_comparison',
        'SQL_sort_order',
        'SQL_NonDB_number',
        'SQL_DB_text',
        'SQL_DB_number',
    ]

    source_db_content_index_paths = {
        'bird-dev': './data/bird-062024/dev/db_contents_index',
        # 'bird-train': './data/bird-062024/train/db_contents_index',
        # 'spider-train': './data/sft_data_collections/spider/db_contents_index',
        # # Add other sources as needed
        # 'dr.spider-DB_schema_synonym': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/DB_schema_synonym/db_contents_index',
        # 'dr.spider-DB_schema_abbreviation': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/DB_schema_abbreviation/db_contents_index',
        # 'dr.spider-DB_DBcontent_equivalence': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/DB_DBcontent_equivalence/db_contents_index',

        # 'dr.spider-NLQ_keyword_synonym': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/NLQ_keyword_synonym/db_contents_index',
        # 'dr.spider-NLQ_keyword_carrier': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/NLQ_keyword_carrier/db_contents_index',
        # 'dr.spider-NLQ_column_synonym': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/NLQ_column_synonym/db_contents_index',
        # 'dr.spider-NLQ_column_carrier': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/NLQ_column_carrier/db_contents_index',
        # 'dr.spider-NLQ_column_attribute': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/NLQ_column_attribute/db_contents_index',
        # 'dr.spider-NLQ_column_value': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/NLQ_column_value/db_contents_index',
        # 'dr.spider-NLQ_value_synonym': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/NLQ_value_synonym/db_contents_index',
        # 'dr.spider-NLQ_multitype': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/NLQ_multitype/db_contents_index',
        # 'dr.spider-NLQ_others': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/NLQ_others/db_contents_index',
        
        # 'dr.spider-SQL_comparison': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/SQL_comparison/db_contents_index',
        # 'dr.spider-SQL_sort_order': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/SQL_sort_order/db_contents_index',
        # 'dr.spider-SQL_DB_text': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/SQL_DB_text/db_contents_index',
        # 'dr.spider-SQL_DB_number': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/SQL_DB_number/db_contents_index',
        
        # 'dr.spider-SQL_NonDB_number': './data/sft_data_collections/diagnostic-robustness-text-to-sql/data/SQL_NonDB_number/db_contents_index',



        # "bank_financials-dev": 'data/sft_data_collections/domain_datasets/db_contents_index/',
    }
    synonym_sources = {
        # 'spider-dev': 'spider-train',
        # 'spider-dk': 'spider-train',
        # 'spider-syn-dev': 'spider-train',
        # 'spider-realistic': 'spider-train',

        # 'bank_financials-train': 'bank_financials-dev',
        # 'aminer_simplified-train': 'bank_financials-dev',
        # 'aminer_simplified-dev': 'bank_financials-dev',
    }

    if args.db_content_index is not None:
        # delete all the default paths except the one specified
        source_db_content_index_paths = {k: v for k, v in source_db_content_index_paths.items() if k == args.db_content_index}

    column_content_searcher = ColumnContentSearcher(source_db_content_index_paths, synonym_sources)

# Request model
class SearchRequest(BaseModel):
    source: str
    db_id: str
    table: str
    column: str
    query: str
    k: Optional[int] = 10  # Number of results to return

# Response model
class SearchResponse(BaseModel):
    results: List[str]

@app.post("/search_column_content", response_model=SearchResponse)
def search_column_content(request: SearchRequest):
    if column_content_searcher is None:
        raise HTTPException(status_code=500, detail="Searcher not initialized.")
    results = column_content_searcher.search_column_content(
        source=request.source,
        db_id=request.db_id,
        table=request.table,
        column=request.column,
        query=request.query,
        k=request.k
    )
    if results is None:
        print(request.source, request.db_id, request.table, request.column)
        raise HTTPException(status_code=404, detail="Searcher not found for the given db_id, table, and column.")
    return SearchResponse(results=results)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lazy_load", action='store_true')
    parser.add_argument("--port", type=int, default=8005)
    parser.add_argument("--db_content_index", type=str, default=None)
    args = parser.parse_args()

    # Run the app with Uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
