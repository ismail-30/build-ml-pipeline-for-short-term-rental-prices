#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import pandas as pd
import os
import argparse
import logging
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    # Creating instance
    run = wandb.init(
        project='nyc_airbnb',
        group='dev',
        job_type="basic_cleaning")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info("Creating dataframe")
    df = pd.read_csv(artifact_local_path)

    logger.info("Dropping outliers")
    idx = df['price'].between(args.min_price, args.max_price)

    logger.info("Copying dataframe")
    df = df[idx].copy()

    logger.info("Converting date into datetime format")
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("Saving dataframe as a csv file")
    df.to_csv("clean_sample.csv", index=False)

    logger.info("Saving dataframe as a csv file")
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file("clean_sample.csv")
    run.log_artifact(artifact)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step cleans the data")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help='Fully qualified name for the input artifact',
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help='Name of the output artifact',
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help='Type of the output artifact',
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help='Description of the output artifact',
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help='Minimum price in the dataset',
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help='Maximum price in the dataset',
        required=True
    )

    args = parser.parse_args()

    go(args)