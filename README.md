# VAD Analysis

This repository provides tools and scripts for performing Voice Activity Detection (VAD) analysis. It includes configurations, models, and utilities to process audio files and detect speech segments.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features

- Load and process audio files for VAD analysis.
- Utilize pre-trained VAD models for accurate speech detection.
- Customize configurations to suit different analysis needs.

## Architecture
![alt text](assets/architecture.png)

## Requirements

Ensure you have the following installed:

- Python==3.10
- [PyTorch](https://pytorch.org/get-started/locally/)
- Additional dependencies as listed in `environment.yml`

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/sktlim/vad-analysis.git
   ```


2. **Navigate to the Project Directory**:

   ```bash
   cd vad-analysis
   ```


3. **Set Up the Environment**:

   It's recommended to use [Anaconda](https://www.anaconda.com/products/distribution) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies. Create a new environment using the provided `environment.yml`:

   ```bash
   conda env create -f environment.yml
   ```


4. **Activate the Environment**:

   ```bash
   conda activate vad-analysis
   ```


## Usage

1. **Prepare Your Audio Data**:

   Place your input audio files in the designated directory as specified in your configuration.

2. **Configure Settings**:

   Create your own config file named `vad_config.yaml` in the `configs` directory similar to that in `sample_config.yaml` and adjust the parameters to match your analysis requirements.

3. **Run the Main Script**:

   Execute the primary analysis script:

   ```bash
   python -m main
   ```


   This will process the audio files and output the VAD results as specified in your configurations.

## Project Structure

- `configs/`: Contains configuration files for setting up analysis parameters.
- `vad_models/`: Includes pre-trained VAD models and related scripts.
- `main.py`: The main script to initiate VAD analysis.
- `utils.py`: Utility functions to support the analysis process.
- `Dockerfile`: Instructions to create a Docker container for the project.
- `environment.yml`: Lists the dependencies required for the project.
