# seq_fetch.py
from pyfaidx import Fasta

GENOME_PATH = "./api/hg38.fa"
fa = Fasta(GENOME_PATH) 

def get_seq(chrom: str, start: int, end: int) -> str:
    s = fa[chrom][start:end].seq.upper()
    assert len(s) == end - start, (len(s), end - start)
    return s