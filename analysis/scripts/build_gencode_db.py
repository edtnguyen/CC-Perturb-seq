#!/usr/bin/env python3
"""
Build gffutils database from GENCODE GTF file.

This script creates a sqlite3 database from a GENCODE GTF file using gffutils.
This database is required for the perturbation analysis pipeline to annotate genes
with metadata (symbol, type, etc.).

Usage:
    python build_gencode_db.py --gtf path/to/gencode.v46.annotation.gtf --output path/to/gencode.v46.annotation.db

Requirements:
    - gffutils
    - tqdm (optional, for progress bar)
"""

import argparse
import sys
import os
import gffutils
import logging
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Build gffutils database from GENCODE GTF.")
    parser.add_argument("--gtf", required=True, help="Path to input GENCODE GTF file")
    parser.add_argument("--output", required=True, help="Path to output .db file")
    parser.add_argument("--force", action="store_true", help="Overwrite existing database if it exists")
    parser.add_argument("--disable-infer-genes", action="store_true", default=True, 
                        help="Disable gene inference (default: True for GENCODE which has explicit gene features)")
    parser.add_argument("--disable-infer-transcripts", action="store_true", default=True,
                        help="Disable transcript inference (default: True for GENCODE which has explicit transcript features)")
    return parser.parse_args()

def build_database(gtf_path, db_path, force=False, infer_genes=False, infer_transcripts=False):
    """
    Build gffutils database.
    
    Args:
        gtf_path: Path to GTF file
        db_path: Path to output database file
        force: Overwrite existing database
        infer_genes: Infer gene features if missing (False for GENCODE)
        infer_transcripts: Infer transcript features if missing (False for GENCODE)
    """
    if not os.path.exists(gtf_path):
        logger.error(f"Input GTF file not found: {gtf_path}")
        sys.exit(1)
        
    if os.path.exists(db_path) and not force:
        logger.error(f"Output database already exists: {db_path}")
        logger.info("Use --force to overwrite.")
        sys.exit(1)
        
    if os.path.exists(db_path) and force:
        logger.info(f"Removing existing database: {db_path}")
        os.remove(db_path)
        
    logger.info(f"Building database from {gtf_path}...")
    logger.info(f"Output: {db_path}")
    
    try:
        # Create database
        # merge_strategy='merge' handles duplicate entries if any
        # disable_infer_* options are important for GENCODE to avoid unnecessary processing
        db = gffutils.create_db(
            gtf_path, 
            db_path, 
            force=force, 
            keep_order=True, 
            merge_strategy='merge', 
            sort_attribute_values=True,
            disable_infer_genes=not infer_genes, 
            disable_infer_transcripts=not infer_transcripts,
            verbose=True
        )
        
        logger.info("Database creation complete!")
        
    except Exception as e:
        logger.error(f"Failed to create database: {e}")
        logger.error(traceback.format_exc())
        # Clean up partial file
        if os.path.exists(db_path):
            os.remove(db_path)
        sys.exit(1)

    try:
        # Validation
        logger.info("Validating database...")
        # Note: gffutils features_of_type doesn't support integer limit argument
        # We need to manually iterate
        genes_iter = db.features_of_type('gene')
        genes = []
        for i, gene in enumerate(genes_iter):
            if i >= 5:
                break
            genes.append(gene)
            
        if len(genes) > 0:
            logger.info("Successfully retrieved sample gene features:")
            for gene in genes:
                gene_name = gene.attributes.get('gene_name', ['N/A'])[0]
                gene_type = gene.attributes.get('gene_type', ['N/A'])[0]
                logger.info(f"  - {gene.id}: {gene_name} ({gene_type})")
        else:
            logger.warning("No 'gene' features found in database. Check input file format.")
            
    except Exception as e:
        logger.warning(f"Database created, but validation failed: {e}")
        logger.warning(traceback.format_exc())
        # Do not delete the database if ONLY validation fails

def main():
    args = parse_args()
    
    # For GENCODE, we typically want these disabled as the GTF already contains them
    # and inferring them can be slow or incorrect
    build_database(
        args.gtf, 
        args.output, 
        force=args.force,
        infer_genes=not args.disable_infer_genes,
        infer_transcripts=not args.disable_infer_transcripts
    )

if __name__ == "__main__":
    main()
