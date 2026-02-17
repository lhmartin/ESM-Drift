import gzip
import os

from esm_drift.data.extract import sequence_from_structure
from esm_drift.utils import StructureDecoder
from esm_drift.data.extract import EmbeddingExtractor


def run_test_extract_and_decode():

    extractor = EmbeddingExtractor()
    # Example input: a single sequence of length 10
    data_folder = '/home/luke/code/structure-dataset-builder/data/pdb/aa/' 
    file = os.listdir(data_folder)[0]
    path = os.path.join(data_folder, file)


    with gzip.open(path, 'rt') as f:
        content = f.read()

    # save the unzipped content to a new file
    unzipped_file = file[:-3]  # remove the .gz extension
    with open(os.path.join(data_folder, unzipped_file), 'w') as f:
        f.write(content)

    import pathlib
    seq = sequence_from_structure(pathlib.Path(os.path.join(data_folder, unzipped_file)))
    data_dict = extractor.extract(seq[0][1])
    # Now decode the embeddings back to a structure
    decoder = StructureDecoder()
    decoder.save_pdb(data_dict['s_s'], 'test_0.pdb')

if __name__ == "__main__":
    run_test_extract_and_decode()