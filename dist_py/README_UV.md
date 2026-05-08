# Running BeachFX-CSHORE with uv

This project has been configured to run with `uv`.

## Prerequisites

- [uv](https://github.com/astral-sh/uv) installed.

## Dependencies

The following Python dependencies are required and managed by `uv`:
- numpy
- h5py

## Running with Docker (Recommended for macOS)

Since the CSHORE binary is a Linux ELF, the easiest way to run it on macOS is using Docker.

1. **Build the image**:
   ```bash
   docker compose build
   ```

2. **Run the full workflow**:
   ```bash
   docker compose up
   ```

This will run all three steps in sequence and the output files will be available on your host machine in the `dist_py/work/` directory.

## Running Locally (Linux/Windows)

The workflow consists of three steps:

1. **Generate Infiles**:
   ```bash
   uv run dist_py/1_make_cshore_infiles.py
   ```

2. **Run CSHORE**:
   ```bash
   uv run dist_py/2_run_cshore.py
   ```

3. **Generate DAT File**:
   ```bash
   uv run dist_py/3_make_dat_file.py
   ```

## Configuration

User inputs can be modified in the "BEGIN USER INPUT" section of `dist_py/1_make_cshore_infiles.py`.
