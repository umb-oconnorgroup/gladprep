from functools import partial
from joblib import load
from multiprocessing import Pool
import os

from cyvcf2 import VCF
import numpy as np


DATA_DIR = "/app/data"
VARIANTS_PATH = "/app/variants.npz"
IMPUTER_PATH = "/app/imputer.joblib"
SCALER_PATH = "/app/scaler.joblib"
PCA_PATH = "/app/principal_components.npy"
QUERY_PATH = "/app/query.npz"


# assumes all vcf have same samples
def process_vcf(vcf_file, reverse_index, alleles):
    vcf = VCF(vcf_file)
    indices = []
    data = []
    for variant in vcf:
        if not variant.is_snp:
            continue
        if variant.CHROM not in reverse_index or variant.POS not in reverse_index[variant.CHROM]:
            continue
        i = reverse_index[variant.CHROM][variant.POS]
        if variant.REF != alleles[i, 0] or len(variant.ALT) != 1 or variant.ALT[0] != alleles[i, 1]:
            continue
        indices.append(i)
        data.append(np.array([genotypes[:2] for genotypes in variant.genotypes], dtype=float).sum(-1))
    return indices, data

def main():
    files = os.listdir(DATA_DIR)
    vcf_files = [os.path.join(DATA_DIR, f) for f in files if f.endswith("vcf.gz") or f.endswith("vcf")]
    variants = np.load(VARIANTS_PATH)
    chromosomes, positions, alleles = variants["chromosomes"], variants["positions"], variants["alleles"]
    reverse_index = {}
    for i in range(len(positions)):
        if chromosomes[i] not in reverse_index:
            reverse_index[chromosomes[i]] = {}
        reverse_index[chromosomes[i]][positions[i]] = i
    with Pool() as p:
        vcf_data = p.map(partial(process_vcf, reverse_index=reverse_index, alleles=alleles), vcf_files)
    data = None
    for indices, genotypes in vcf_data:
        if data is None:
            data = np.empty((len(genotypes[0]), len(positions)), dtype=float)
            data[:] = np.nan
        data[:, indices] = np.array(genotypes).T
    
    imputer = load(IMPUTER_PATH)
    data = imputer.transform(data)
    genotype_counts = np.zeros((data.shape[1], 3))
    for i in range(0, 3):
        genotype_counts[:, i] = (data == i).sum(0)

    scaler = load(SCALER_PATH)
    data = scaler.transform(data)
    principal_components = np.load(PCA_PATH)
    pca_embedding = np.matmul(data, principal_components)
    np.savez_compressed(QUERY_PATH, pca=pca_embedding, gc=genotype_counts)

if __name__ == "__main__":
    main()
