import pandas as pd
import argparse
import os
import sys
from tqdm import tqdm
from cyvcf2 import VCF

def parse_clinvar_vcf(input_vcf_gz, output_dir):
    """
    Parses a gzipped ClinVar VCF file and saves data chunks as separate
    Parquet files in an output directory.
    """
    print(f"Starting VCF parsing of {input_vcf_gz} with cyvcf2...")
    
    vcf_reader = VCF(input_vcf_gz)
    
    data_list = []
    chunk_size = 100000
    chunk_counter = 0

    for record in tqdm(vcf_reader, desc="Parsing VCF records"):
        clnsig = record.INFO.get('CLNSIG')
        clnrevstat = record.INFO.get('CLNREVSTAT')
        gene_info = record.INFO.get('GENEINFO')
        mc_info = record.INFO.get('MC')

        variant_type = mc_info.split('|')[-1] if mc_info else None
        gene_symbol = gene_info.split(':')[0] if gene_info else None
        alt_allele = record.ALT[0] if record.ALT else None

        data_list.append({
            'Chromosome': record.CHROM,
            'Position': record.POS,
            'RefAllele': record.REF,
            'AltAllele': alt_allele,
            'ClinicalSignificance': clnsig,
            'ReviewStatus': clnrevstat,
            'GeneSymbol': gene_symbol,
            'VariantType': variant_type
        })

        if len(data_list) >= chunk_size:
            df_chunk = pd.DataFrame(data_list)
            # name for chunk
            chunk_filename = os.path.join(output_dir, f'chunk_{chunk_counter}.parquet')
            df_chunk.to_parquet(chunk_filename, engine='pyarrow')
            
            data_list = []
            chunk_counter += 1

    # Write final chunk if any data is left
    if data_list:
        df_chunk = pd.DataFrame(data_list)
        chunk_filename = os.path.join(output_dir, f'chunk_{chunk_counter}.parquet')
        df_chunk.to_parquet(chunk_filename, engine='pyarrow')

    print(f"\nFinished VCF parsing. Data chunks saved to directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse ClinVar VCF to a directory of Parquet files using cyvcf2.")
    parser.add_argument("input_file", help="Path to the input .vcf.gz file.")
    parser.add_argument("output_dir", help="Path for the output directory to store Parquet chunks.")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        if os.listdir(args.output_dir):
            print(f"Error: Output directory '{args.output_dir}' is not empty. Please clear it or choose a different name.")
            sys.exit(1)

    parse_clinvar_vcf(args.input_file, args.output_dir)