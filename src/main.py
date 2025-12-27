from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import MetadataMode
import pprint
from utils import set_utc_log


def main():
    logger= set_utc_log()
    docs= SimpleDirectoryReader(input_dir= './data', filename_as_id= True).load_data()
    logger.info(len(docs))
    logger.info(docs[0].get_content(metadata_mode=MetadataMode.EMBED))
    
    


if __name__ == "__main__":
    main()
