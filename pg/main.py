#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
import torch
import click
from vllm import LLM

from datasets import load_dataset
from config import Config
from rm import load_prm
from search import particle_gibbs
from search import particle_gibbs_batch
from utils.data import get_dataset, save_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os
# Check CUDA_DEVICE_MAX_CONNECTIONS environment variable
logger.info(f"CUDA_DEVICE_MAX_CONNECTIONS: {os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')}")

@click.command()
@click.option(
    "--dataset-start",
    default=0,
    type=int,
    help="Start index of the dataset to process.",
    show_default=True,
)
@click.option(
    "--dataset-end",
    default=38,
    type=int,
    help="End index of the dataset to process.",
    show_default=True,
)
@click.option(
    "--seed",
    default=None,
    # type=int,
    help="Random seed for reproducibility.",
    show_default=True,
)
@click.option(
    "--total-timesteps",
    default=1,
    type=int,
    help="Total timesteps for particle Gibbs sampling.",
    show_default=True,
)
@click.option(
    "--n-particles",
    default=4,
    type=int,
    help="Number of particles for Gibbs sampling.",
    show_default=True,
)
@click.option(
    "--softmax-temp",
    default=1.0,
    type=float,
    help="Softmax temperature for sampling.",
    show_default=True,
)
@click.option(
    "--llm-sampling-temp",
    default=0.8, 
    type=float,
    help="Temperature for LLM sampling.",
    show_default=True,
)
@click.option(
    "--temperature-annealing",
    default=None,
    type=(float, float, int),
    help="Parameters for temperature annealing (start_temp, end_temp, total_steps).",
)
@click.option(
    "--resample-inactive",
    is_flag=True,
    default=False,
    help="Whether to resample inactive particles.",
    show_default=True,
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, writable=True),
    help="Output directory to save the results.",
)
@click.option(
    "--model-path",
    default="meta-llama/Llama-3.2-1B-Instruct",
    type=str,
    help="Path to the language model.",
    show_default=True,
)
@click.option(
    "--prm-path",
    default="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
    type=str,
    help="Path to the probabilistic reward model.",
    show_default=True,
)
@click.option(
    "--dataset-path",
    default=None,
    type=str,
    help="Path to the dataset.",
    show_default=True,
)
def main(
    dataset_start,
    dataset_end,
    seed,
    total_timesteps,
    n_particles,
    llm_sampling_temp,
    softmax_temp,
    temperature_annealing,
    resample_inactive,
    output_dir,
    model_path,
    prm_path,
    dataset_path,
):
    """
    Run Particle Gibbs sampling for a dataset using a specified LLM and reward model.
    """
    start_time = time.time()  # Start timer

    enable_prefix_caching = False

    # Log all the arguments
    logger.info("Starting execution with the following parameters:")
    logger.info(f"Dataset start: {dataset_start}")
    logger.info(f"Dataset end: {dataset_end}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Number of particles: {n_particles}")
    logger.info(f"LLM sampling temperature: {llm_sampling_temp}")
    logger.info(f"Softmax temperature: {softmax_temp}")
    logger.info(f"Temperature annealing: {temperature_annealing}")
    logger.info(f"Resample inactive: {resample_inactive}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Enable prefix caching: {enable_prefix_caching}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"PRM path: {prm_path}")
    logger.info(f"Dataset path: {dataset_path}")

    config = Config()
    config.output_dir = output_dir
    config.model_path = model_path
    config.prm_path = prm_path

    # Initialize LLM with available GPUs
    num_gpus = 4 if "qwen2" in config.model_path.lower() else torch.cuda.device_count()

    if seed == None or seed == "None":
        logger.info("Initializing LLM without seed")
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            tensor_parallel_size=num_gpus,
        )
    else:
        seed = int(seed)
        logger.info(f"Initializing LLM with seed: {seed}")
        llm = LLM(
            model=config.model_path,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enable_prefix_caching=enable_prefix_caching,
            seed=seed,
            tensor_parallel_size=num_gpus,
        )

    # Load the probabilistic reward model
    prm = load_prm(config)

    # Load and preprocess the dataset
    if dataset_path is not None:
        dataset = load_dataset("json", data_files=dataset_path, split="train")
    else:
        dataset = get_dataset(config)

    dataset = dataset.select(range(dataset_start, dataset_end))
    # Filter the dataset to include only ones not already completed in the directory.
    if output_dir is not None:
        logger.info(f"Filtering dataset to exclude already completed problems in {output_dir}")
        if isinstance(dataset[0]['unique_id'], int):
            dataset = dataset.filter(lambda x: not os.path.exists(os.path.join(output_dir, f"{x['unique_id']}.pkl")))
        else:
            dataset = dataset.filter(lambda x: not os.path.exists(os.path.join(output_dir, f"{x['unique_id'].replace('/', '_')[:-5]}.pkl")))
        

    logger.info(f"Length of dataset: {len(dataset)}")

    # Perform Particle Gibbs sampling on the dataset
    dataset = dataset.map(
        particle_gibbs_batch if config.use_continuous_batching else particle_gibbs,
        batched=False,
        batch_size=1,
        fn_kwargs={
            "config": config,
            "llm": llm,
            "prm": prm,
            "total_timesteps": total_timesteps,
            "n_particles": n_particles,
            "llm_sampling_temp": llm_sampling_temp,
            "softmax_temp": softmax_temp,
            "temperature_annealing": temperature_annealing,
            "resample_inactive": resample_inactive,
        },
        desc="Running search",
        load_from_cache_file=False,
    )

    end_time = time.time()  # End timer
    elapsed_time = end_time - start_time
    logger.info(f"Done 🔥! Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()