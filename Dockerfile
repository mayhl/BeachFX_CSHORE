# Stage 1: Get libgfortran4 and dependencies from Ubuntu 20.04
FROM ubuntu:20.04 AS libs
RUN apt-get update && apt-get install -y libgfortran4

# Stage 2: Final image
FROM python:3.12-slim

# Copy the libraries from the libs stage
COPY --from=libs /usr/lib/x86_64-linux-gnu/libgfortran.so.4* /usr/lib/x86_64-linux-gnu/
COPY --from=libs /usr/lib/x86_64-linux-gnu/libquadmath.so.0* /usr/lib/x86_64-linux-gnu/

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy the project files
COPY . .

# Ensure the Linux binary is executable
RUN chmod +x dist_py/executables/CSHORE_USACE_LINUX.out

# Install dependencies using uv
RUN uv sync

# Default command to run the workflow
CMD ["uv", "run", "dist_py/1_make_cshore_infiles.py"]
