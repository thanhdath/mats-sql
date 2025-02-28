from utils.db_utils import get_cursor_from_path, execute_sql_long_time_limitation
import json
import os, shutil
from tqdm import tqdm

def remove_contents_of_a_folder(index_path):
    # if index_path does not exist, then create it
    os.makedirs(index_path, exist_ok = True)
    # remove files in index_path
    for filename in os.listdir(index_path):
        file_path = os.path.join(index_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def lowercase_directory_name(directory_path):
    # Get the absolute path of the directory
    abs_directory_path = os.path.abspath(directory_path)
    
    # Split the directory path to get the parent directory and the folder name
    parent_dir, current_dir = os.path.split(abs_directory_path)
    
    # Convert the folder name to lowercase
    lowercase_dir = current_dir.lower()
    
    # Define the new directory path with the lowercase folder name
    new_directory_path = os.path.join(parent_dir, lowercase_dir).lower()

    # if os.path.isdir(abs_directory_path) and os.path.isdir(new_directory_path):
    #     shutil.rmtree(abs_directory_path)
    #     return
    
    # Rename the directory
    if current_dir != lowercase_dir:
        os.rename(abs_directory_path, new_directory_path)
        print(f"Renamed directory: '{current_dir}' to '{lowercase_dir}'")
    else:
        print(f"The directory '{current_dir}' is already in lowercase.")

def build_content_index(db_path, index_path):
    '''
    Create a BM25 index for all contents in a database
    '''
    cursor = get_cursor_from_path(db_path)
    results = execute_sql_long_time_limitation(cursor, "SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [result[0] for result in results]
    
    for table_name in table_names:
        # skip SQLite system table: sqlite_sequence
        table_name = table_name.lower()
        if table_name == "sqlite_sequence":
            continue
        results = execute_sql_long_time_limitation(cursor, "SELECT name FROM PRAGMA_TABLE_INFO('{}')".format(table_name))
        column_names_in_one_table = [result[0] for result in results]
        for column_name in column_names_in_one_table:
            column_name = column_name.lower()
            column_index_path = f"{index_path}/{table_name}-**-{column_name}"

            if os.path.exists(column_index_path):
                # lowercase folder directory
                # lowercase_directory_name(column_index_path)
                continue
        

            all_column_contents = []
            try:
                print("SELECT DISTINCT `{}` FROM `{}` WHERE `{}` IS NOT NULL;".format(column_name, table_name, column_name))
                results = execute_sql_long_time_limitation(cursor, "SELECT DISTINCT `{}` FROM `{}` WHERE `{}` IS NOT NULL;".format(column_name, table_name, column_name))
                column_contents = [str(result[0]).strip() for result in results]

                for c_id, column_content in enumerate(column_contents):
                    # remove empty and extremely-long contents
                    if len(column_content) != 0 and len(column_content) <= 50:
                        all_column_contents.append(
                            {
                                "id": "{}-**-{}-**-{}".format(table_name, column_name, c_id).lower(),
                                "contents": column_content
                            }
                        )
            except Exception as e:
                print(str(e))

            os.makedirs('./data/temp_db_index', exist_ok = True)
            
            with open("./data/temp_db_index/contents.json", "w") as f:
                f.write(json.dumps(all_column_contents, indent = 2, ensure_ascii = True))

            # Building a BM25 Index (Direct Java Implementation), see https://github.com/castorini/pyserini/blob/master/docs/usage-index.md
            cmd = "python -m pyserini.index.lucene --collection JsonCollection --input ./data/temp_db_index --index \"{}\" --generator DefaultLuceneDocumentGenerator --threads 16 --storePositions --storeDocvectors --storeRaw".format(column_index_path)
            
            print(f"added {column_index_path}")
            d = os.system(cmd)
            print(d)
            os.remove("./data/temp_db_index/contents.json")

if __name__ == "__main__":
    # print("build content index for BIRD's training set databases...")
    # remove_contents_of_a_folder("data/bird-062024/train/db_contents_index")
    # # build content index for BIRD's training set databases
    # for db_id in tqdm(os.listdir("data/bird-062024/train/train_databases")):
    #     if db_id.endswith(".json"):
    #         continue
    #     print(db_id)
    #     build_content_index(
    #         os.path.join("data/bird-062024/train/train_databases/", db_id, db_id + ".sqlite"),
    #         os.path.join("data/bird-062024/train/db_contents_index/", db_id)
    #     )
    
    # print("build content index for BIRD's dev set databases...")
    # remove_contents_of_a_folder("data/bird-062024/dev/db_contents_index")
    # # build content index for BIRD's dev set databases
    # for db_id in tqdm(os.listdir("data/bird-062024/dev/dev_databases")):
    #     print(db_id)
    #     build_content_index(
    #         os.path.join("data/bird-062024/dev/dev_databases/", db_id, db_id + ".sqlite"),
    #         os.path.join("data/bird-062024/dev/db_contents_index/", db_id)
    #     )


    # print("build content index for BIRD's training set databases...")
    # remove_contents_of_a_folder("data/sft_data_collections/bird/train/db_contents_index")
    # # build content index for BIRD's training set databases
    # for db_id in tqdm(os.listdir("data/sft_data_collections/bird/train/train_databases")):
    #     if db_id.endswith(".json"):
    #         continue
    #     print(db_id)
    #     build_content_index(
    #         os.path.join("data/sft_data_collections/bird/train/train_databases/", db_id, db_id + ".sqlite"),
    #         os.path.join("data/sft_data_collections/bird/train/db_contents_index/", db_id)
    #     )
    
    # print("build content index for BIRD's dev set databases...")
    # remove_contents_of_a_folder("data/sft_data_collections/bird/dev/db_contents_index")
    # # build content index for BIRD's dev set databases
    # for db_id in tqdm(os.listdir("data/sft_data_collections/bird/dev/dev_databases")):
    #     print(db_id)
    #     build_content_index(
    #         os.path.join("data/sft_data_collections/bird/dev/dev_databases/", db_id, db_id + ".sqlite"),
    #         os.path.join("data/sft_data_collections/bird/dev/db_contents_index/", db_id)
    #     )



    # print("build content index for spider's databases...")
    # remove_contents_of_a_folder("./data/sft_data_collections/spider/db_contents_index")
    # # build content index for spider's databases
    # for db_id in os.listdir("./data/sft_data_collections/spider/database"):
    #     print(db_id)
    #     build_content_index(
    #         os.path.join("./data/sft_data_collections/spider/database/", db_id, db_id + ".sqlite"),
    #         os.path.join("./data/sft_data_collections/spider/db_contents_index/", db_id)
    #     )
    
    # print("build content index for Dr.Spider's 17 perturbation test sets...")
    # # build content index for Dr.Spider's 17 perturbation test sets
    # test_set_names = os.listdir("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data")
    # test_set_names.remove("Spider-dev")
    # for test_set_name in test_set_names:
    #     if test_set_name.startswith("DB_"):
    #         remove_contents_of_a_folder(os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "db_contents_index"))
    #         for db_id in os.listdir(os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "database_post_perturbation")):
    #             print(db_id)
    #             build_content_index(
    #                 os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "database_post_perturbation", db_id, db_id + ".sqlite"),
    #                 os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "db_contents_index", db_id)
    #             )
    #     else:
    #         remove_contents_of_a_folder(os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "db_contents_index"))
    #         for db_id in os.listdir(os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "databases")):
    #             if db_id in ["README.md", "database_original"]:
    #                 continue
    #             print(db_id)
    #             build_content_index(
    #                 os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "databases", db_id, db_id + ".sqlite"),
    #                 os.path.join("./data/sft_data_collections/diagnostic-robustness-text-to-sql/data/", test_set_name, "db_contents_index", db_id)
    #             )
    
    # print("build content index for Bank_Financials and Aminer_Simplified training set databases...")
    # remove_contents_of_a_folder("./data/sft_data_collections/domain_datasets/db_contents_index")
    # # build content index for Bank_Financials's training set databases
    # for db_id in os.listdir("./data/sft_data_collections/domain_datasets/databases"):
    #     print(db_id)
    #     build_content_index(
    #         os.path.join("./data/sft_data_collections/domain_datasets/databases/", db_id, db_id + ".sqlite"),
    #         os.path.join("./data/sft_data_collections/domain_datasets/db_contents_index/", db_id)
    #     )


    print("build content index for MACSQL spider's databases...")
    remove_contents_of_a_folder("/home/datht/llmsql/data//spider/db_contents_index")
    # build content index for spider's databases
    for db_id in os.listdir("/home/datht/llmsql/data/spider/database"):
        print(db_id)
        build_content_index(
            os.path.join("/home/datht/llmsql/data/spider/database", db_id, db_id + ".sqlite"),
            os.path.join("/home/datht/llmsql/data//spider/db_contents_index", db_id)
        )