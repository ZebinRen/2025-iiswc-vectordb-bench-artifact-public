# 2025 IISWC VectorDB Bench Artifact

## Setup Environment

### Install BPF trace

Install from apt:

```bash
sudo apt-get update
sudo apt-get install bpftrace
```

```bash
sudo apt-get install asciidoctor binutils-dev bison build-essential clang cmake flex libbpf-dev libbpfcc-dev libcereal-dev libdw-dev libelf-dev libiberty-dev libpcap-dev llvm-dev libclang-dev systemtap-sdt-dev zlib1g-dev
sudo apt install llvm-16-dev clang-16 libclang-16-dev libpolly-16-dev
mkdir local
cd ~/local
git clone https://github.com/iovisor/bpftrace.git
cd bpftrace
git checkout v0.19.0
git submodule init
git submodule update
export LLVM_ROOT=/usr/lib/llvm-16  
export CC=clang-16 CXX=clang++-16
cmake -DCMAKE_INSTALL_PREFIX='/home/zebin/local/bpftrace/build-libs' \
      -DCMAKE_PREFIX_PATH='/home/zebin/local/bpftrace/build-libs' \
      -DCMAKE_USE_LIBBPF_PACKAGE=ON -DCMAKE_INSTALL_LIBDIR=lib64 \
      -DENABLE_MAN=0 -DENABLE_EXAMPLES=0 -DENABLE_TESTS=0 \
      -DENABLE_LIBDEBUGINFOD=0 -DLUAJIT= -DENABLE_LLVM_NATIVECODEGEN=0 \
      ./bcc
mkdir build && cd build
../build-libs.sh
cd ..
cmake -B ./build -DBUILD_TESTING=OFF
make -C ./build -j$(nproc)
```

The binary file is located in //bpftrace/build/src/bpftrace, this path is needed with the BPFTRACE_BIN environment variable.


### Install Python Environment and VectorDBBench:

```bash
# Install anconada
wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh 
./Anaconda3-2024.02-1-Linux-x86_64.sh # follow the install procedure
# reopen the terminal to activate the conda environment
conda create --name vectordb-bench python=3.11.10
conda activate vectordb-bench
conda install pip

# Get the location of the python binary, we will need this later for the experiments with I/O traces
# We will refer to this path as `PYTHON_BIN` in the following sections
whereis python /home/zebin/anaconda3/envs/vectordb-bench/bin/python
>> /home/zebin/anaconda3/envs/vectordb-bench-new/bin/python  # Example output

# Install VectorDBBench and dependencies
git clone https://github.com/ZebinRen/VectorDBBench-dev.git
cd VectorDBBench-dev
git checkout origin/benchmark-paper
pip install -e .

# Install the dependencies
pip install vectordb-bench[qdrant]
pip install vectordb-bench[weaviate]
pip install lancedb pandas
```

### Install other packaged needed

[docker](https://docs.docker.com/engine/install/ubuntu/)

## Set Up VectorDBs and Prepare the Indexes

Clone this repository:

```bash
git clone https://github.com/ZebinRen/2025-iiswc-vectordb-bench-artifact-public
```

This artifact can also be accessed via Zenodo: [https://doi.org/10.5281/zenodo.16916496](https://doi.org/10.5281/zenodo.16916496).

We use a dedicated NVMe SSD to store the database, such as the vector data and indexes, to avoid the interference from other workloads such as the operating system.
Before building the indexes, create the root directory for all the databases.
We use ```DATA_ROOT``` to refer to the root directory of the database.

Note: build the indexes of Cohere-10M and OpenAI-5M may take one or two days, depending on the CPU performance.

Before building the indexes, set the environment variables:

```bash
# To get the PYTHON_BIN, activate the conda environment and run `whereis python`
export PYTHON_BIN=/path/to/your/python/bin/in/the/conda/environment
# To get the VECTORDB_BENCH_BIN, run `whereis vectordbbench` in the conda environment
export VECTORDB_BENCH_BIN=/path/to/your/vectordb-bench/bin/vectordbbench
# The bpftrace binary, which is compiled in the previous step
export BPFTRACE_BIN=/path/to/your/bpftrace/bin/bpftrace
# The directory that stores the database data
export DATA_ROOT=/path/to/your/data/root
```

### Milvus:

```bash
cd ${DATA_ROOT}
# copy the milvus config file
mkdir milvus
cd milvus
mkdir milvus-ivf-cohere-1m milvus-ivf-cohere-10m milvus-ivf-openai-500k milvus-ivf-openai-5m
mkdir milvus-hnsw-cohere-1m milvus-hnsw-cohere-10m milvus-hnsw-openai-500k milvus-hnsw-openai-5m
mkdir milvus-diskann-cohere-1m milvus-diskann-cohere-10m milvus-diskann-openai-500k milvus-diskann-openai-5m

# change the directory to the root directory of this artifact
cp milvus-configs/milvus-ivf-cohere-1m-standalone_embed.sh ${DATA_ROOT}/milvus/milvus-ivf-cohere-1m/
cp milvus-configs/milvus-ivf-cohere-10m-standalone_embed.sh milvus-ivf-cohere-10m/
cp milvus-configs/milvus-ivf-openai-500k-standalone_embed.sh milvus-ivf-openai-500k/
cp milvus-configs/milvus-ivf-openai-5m-standalone_embed.sh milvus-ivf-openai-5m/
cp milvus-configs/milvus-hnsw-cohere-1m-standalone_embed.sh milvus-hnsw-cohere-1m/
cp milvus-configs/milvus-hnsw-cohere-10m-standalone_embed.sh milvus-hnsw-cohere-10m/
cp milvus-configs/milvus-hnsw-openai-500k-standalone_embed.sh milvus-hnsw-openai-500k/
cp milvus-configs/milvus-hnsw-openai-5m-standalone_embed.sh milvus-hnsw-openai-5m/
cp milvus-configs/milvus-diskann-cohere-1m-standalone_embed.sh milvus-diskann-cohere-1m/
cp milvus-configs/milvus-diskann-cohere-10m-standalone_embed.sh milvus-diskann-cohere-10m/
cp milvus-configs/milvus-diskann-openai-500k-standalone_embed.sh milvus-diskann-openai-500k/
cp milvus-configs/milvus-diskann-openai-5m-standalone_embed.sh milvus-diskann-openai-5m/

# start the docker containers, do the same operation for all the configs
cd milvus-ivf-cohere-1m
bash standalone_embed.sh start
cd ../milvus-ivf-cohere-10m
bash standalone_embed.sh start
# We do not enumerate all the commands here for simplicity.
```

```bash
# IVF
nohup vectordbbench milvusivfflat --db-label milvus_ivf_cohere_1m --case-type Performance768D1M --k 10 --num-concurrency 1 --uri http://localhost:19531 --password "" --lists 4000 --probes 30 > milvus-ivf-cohere-1m.txt 2>&1 &
nohup vectordbbench milvusivfflat --db-label milvus_ivf_cohere_10m --case-type Performance768D10M --k 10 --num-concurrency 1 --uri http://localhost:19532 --password "" --lists 12648 --probes 30 > milvus-ivf-cohere-10m.txt 2>&1 &
nohup vectordbbench milvusivfflat --db-label milvus_ivf_openai_500k --case-type Performance1536D500K --k 10 --num-concurrency 1 --uri http://localhost:19533 --password "" --lists 2828  --probes 30 > milvus-ivf-openai-500k.txt 2>&1 &
nohup vectordbbench milvusivfflat --db-label milvus_ivf_openai_5m --case-type Performance1536D5M --k 10 --num-concurrency 1 --uri http://localhost:19534 --password "" --lists 8944 --probes 30 > milvus-ivf-openai-5m.txt 2>&1 &
# HNSW
nohup vectordbbench milvushnsw --db-label milvus_hnsw_cohere_1m --case-type Performance768D1M --k 10 --num-concurrency 1 --uri http://localhost:19535 --password "" --m 16 --ef-construction 200 --ef-search 50 > milvus-hnsw-cohere-1m.txt 2>&1 &
nohup vectordbbench milvushnsw --db-label milvus_hnsw_cohere_10m --case-type Performance768D10M --k 10 --num-concurrency 1 --uri http://localhost:19536 --password "" --m 16 --ef-construction 200 --ef-search 50 > milvus-hnsw-cohere-10m.txt 2>&1 &
nohup vectordbbench milvushnsw --db-label milvus_hnsw_openai_500k --case-type Performance1536D500K --k 10 --num-concurrency 1 --uri http://localhost:19537 --password "" --m 16 --ef-construction 200 --ef-search 50 > milvus-hnsw-openai-500k.txt 2>&1 &
nohup vectordbbench milvushnsw --db-label milvus_hnsw_openai_5m --case-type Performance1536D5M --k 10 --num-concurrency 1 --uri http://localhost:19538 --password "" --m 16 --ef-construction 200 --ef-search 50 > milvus-hnsw-openai-5m.txt 2>&1 &
# DiskANN
nohup vectordbbench milvusdiskann --db-label milvus_diskann_cohere_1m --case-type Performance768D1M --k 10 --num-concurrency 1 --uri http://localhost:19539 --password "" --search-list 20 > milvus-diskann-cohere-1m.txt 2>&1 &
nohup vectordbbench milvusdiskann --db-label milvus_diskann_cohere_10m --case-type Performance768D10M --k 10 --num-concurrency 1 --uri http://localhost:19540 --password "" --search-list 20 > milvus-diskann-cohere-10m.txt 2>&1 &
nohup vectordbbench milvusdiskann --db-label milvus_diskann_openai_500k --case-type Performance1536D500K --k 10 --num-concurrency 1 --uri http://localhost:19541 --password "" --search-list 20 > milvus-diskann-openai-500k.txt 2>&1 &
nohup vectordbbench milvusdiskann --db-label milvus_diskann_openai_5m --case-type Performance1536D5M --k 10 --num-concurrency 1 --uri http://localhost:19542 --password "" --search-list 20 > milvus-diskann-openai-5m.txt 2>&1 &
```

After the indexes are builts, the docker can be stopped to avoid the interference from the other workloads.

```bash
# Example of stop a milvus docker container
cd ${DATA_ROOT}/milvus/milvus-ivf-cohere-1m
bash standalone_embed.sh stop
# We do not enumerate all the commands here for simplicity.
```

### Qdrant

Create the directory for Qdrant:

```bash
cd ${DATA_ROOT}
mkdir qdrant
cd qdrant
mkdir qdrant-hnsw-mem-cohere-1m
mkdir qdrant-hnsw-mem-cohere-10m
mkdir qdrant-hnsw-mem-openai-500k
mkdir qdrant-hnsw-mem-openai-5m
```

Start docker containers

```bash
docker pull qdrant/qdrant:latest
## Start Qdrant docker container
sudo docker run -d --name qdrant-hnsw-mem-cohere-1m -p 6333:6333 -p 6334:6334  -v "${DATA_ROOT}/qdrant/qdrant-hnsw-mem-cohere-1m:/qdrant/storage:z"  qdrant/qdrant
sudo docker run -d --name qdrant-hnsw-mem-cohere-10m -p 6335:6333 -p 6336:6334  -v "${DATA_ROOT}/qdrant/qdrant-hnsw-mem-cohere-10m:/qdrant/storage:z"  qdrant/qdrant
sudo docker run -d --name qdrant-hnsw-mem-openai-500k -p 6337:6333 -p 6338:6334  -v "${DATA_ROOT}/qdrant/qdrant-hnsw-mem-openai-500k:/qdrant/storage:z"  qdrant/qdrant
sudo docker run -d --name qdrant-hnsw-mem-openai-5m -p 6339:6333 -p 6340:6334  -v "${DATA_ROOT}/qdrant/qdrant-hnsw-mem-openai-5m:/qdrant/storage:z"  qdrant/qdrant
```

Build the indexes

```bash
# memory
nohup vectordbbench qdrantlocal --case-type Performance768D1M --k 10 --url "http://localhost:6333" --grpc-port 6334 --m 16 --ef-construct 200 --num-concurrency 1 > qdrant-hnsw-mem-cohere-1m.txt 2>&1 &
nohup vectordbbench qdrantlocal --case-type Performance768D10M --k 10 --url "http://localhost:6335" --grpc-port 6336 --m 16 --ef-construct 200 --num-concurrency 1 > qdrant-hnsw-mem-cohere-10m.txt 2>&1 &
nohup vectordbbench qdrantlocal --case-type Performance1536D500K --k 10 --url "http://localhost:6337" --grpc-port 6338 --m 16 --ef-construct 200 --num-concurrency 1 > qdrant-hnsw-mem-openai-500k.txt 2>&1 &
nohup vectordbbench qdrantlocal --case-type Performance1536D5M --k 10 --url "http://localhost:6339" --grpc-port 6340 --m 16 --ef-construct 200 --num-concurrency 1 > qdrant-hnsw-mem-openai-5M.txt 2>&1 &
```

The docker container of Qdrant can be stopped with docker command.

```bash
# Example of stop a docker container of Qdrant
docker stop qdrant-hnsw-mem-cohere-1m
docker stop qdrant-hnsw-mem-cohere-10m
docker stop qdrant-hnsw-mem-openai-500k
docker stop qdrant-hnsw-mem-openai-5m
```

### Weaviate

Create the directory for Weaviate:

```bash
cd ${DATA_ROOT}
mkdir weaviate
cd weaviate
mkdir weaviate-hnsw-cohere-1m
mkdir weaviate-hnsw-cohere-10m
mkdir weaviate-hnsw-openai-500k
mkdir weaviate-hnsw-openai-5m
```

Start docker containers:

```bash
sudo -E docker run -d --name weaviate-hnsw-cohere-1m -p 8081:8080 -p 50051:50051 -v ${DATA_ROOT}/weaviate/weaviate-hnsw-cohere-1m:/var/lib/weaviate -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" cr.weaviate.io/semitechnologies/weaviate:1.31.0
sudo -E docker run -d --name weaviate-hnsw-cohere-10m -p 8082:8080 -p 50052:50051 -v ${DATA_ROOT}/weaviate/weaviate-hnsw-cohere-10m:/var/lib/weaviate -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" cr.weaviate.io/semitechnologies/weaviate:1.31.0
sudo -E docker run -d --name weaviate-hnsw-openai-500k -p 8083:8080 -p 50053:50051 -v ${DATA_ROOT}/weaviate/weaviate-hnsw-openai-500k:/var/lib/weaviate -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" cr.weaviate.io/semitechnologies/weaviate:1.31.0
sudo -E docker run -d --name weaviate-hnsw-openai-5m -p 8084:8080 -p 50054:50051 -v ${DATA_ROOT}/weaviate/weaviate-hnsw-openai-5m:/var/lib/weaviate -e PERSISTENCE_DATA_PATH="/var/lib/weaviate" cr.weaviate.io/semitechnologies/weaviate:1.31.0
```

Build the indexes

```bash
nohup vectordbbench weaviate --case-type Performance768D1M --k 10  --url "http://localhost:8081" --num-concurrency 1 --no-auth --m 16 --ef-construction 200 --ef 10 > weaviate-hnsw-cohere-1m.txt 2>&1 &
nohup vectordbbench weaviate --case-type Performance768D10M --k 10  --url "http://localhost:8082" --num-concurrency 1 --no-auth --m 16 --ef-construction 200 --ef 10 > weaviate-hnsw-cohere-10m.txt 2>&1 &
nohup vectordbbench weaviate --case-type Performance1536D500K --k 10  --url "http://localhost:8083" --num-concurrency 1 --no-auth --m 16 --ef-construction 200 --ef 10 > weaviate-hnsw-openai-500k.txt 2>&1 &
nohup vectordbbench weaviate --case-type Performance1536D5M --k 10  --url "http://localhost:8084" --num-concurrency 1 --no-auth --m 16 --ef-construction 200 --ef 10 > weaviate-hnsw-openai-5m.txt 2>&1 &
```

The docker container of Weaviate can be stopped with docker command.

```bash
# Example of stop a docker container of Weaviate
docker stop weaviate-hnsw-cohere-1m
docker stop weaviate-hnsw-cohere-10m
docker stop weaviate-hnsw-openai-500k
docker stop weaviate-hnsw-openai-5m
```

### LanceDB

Create the directory for LanceDB:

```bash
cd ${DATA_ROOT}
mkdir lancedb
cd lancedb
mkdir lancedb-ivfpq-cohere-1m
mkdir lancedb-ivfpq-cohere-10m
mkdir lancedb-ivfpq-openai-500k
mkdir lancedb-ivfpq-openai-5m
mkdir lancedb-hnsw-cohere-1m
mkdir lancedb-hnsw-cohere-10m
mkdir lancedb-hnsw-openai-500k
mkdir lancedb-hnsw-openai-5m
```

Build the indexes

```bash
# LanceDB IVFPQ
nohup vectordbbench lancedbivfpq --case-type Performance768D1M --k 10 --num-concurrency 1 --uri ${DATA_ROOT}/lancedb/lancedb-ivfpq-cohere-1m --num-partitions 4000 --nprobes 20 > lancedb-ivfpq-cohere-1m.txt 2>&1 &
nohup vectordbbench lancedbivfpq --case-type Performance768D10M --k 10 --num-concurrency 1 --uri ${DATA_ROOT}/lancedb/lancedb-ivfpq-cohere-10m --num-partitions 12648 --nprobes 20 > lancedb-ivfpq-cohere-10m.txt 2>&1 &
nohup vectordbbench lancedbivfpq --case-type Performance1536D500K --k 10 --num-concurrency 1 --uri ${DATA_ROOT}/lancedb/lancedb-ivfpq-openai-500k --num-partitions 2828 --nprobes 20 > lancedb-ivfpq-openai-500k.txt 2>&1 &
nohup vectordbbench lancedbivfpq --case-type Performance1536D5M --k 10 --num-concurrency 1 --uri ${DATA_ROOT}/lancedb/lancedb-ivfpq-openai-5m --num-partitions 8944 --nprobes 20 > lancedb-ivfpq-openai-5m.txt 2>&1 &

# LanceDB HNSW
nohup vectordbbench lancedbhnsw --case-type Performance768D1M --k 10 --num-concurrency 1 --uri ${DATA_ROOT}/lancedb/lancedb-hnsw-cohere-1m --m 16 --ef-construction 200 --ef 50 > lancedb-hnsw-cohere-1m.txt 2>&1 &
nohup vectordbbench lancedbhnsw --case-type Performance768D10M --k 10 --num-concurrency 1 --uri ${DATA_ROOT}/lancedb/lancedb-hnsw-cohere-10m --m 16 --ef-construction 200 --ef 50 > lancedb-hnsw-cohere-10m.txt 2>&1 &
nohup vectordbbench lancedbhnsw --case-type Performance1536D500K --k 10 --num-concurrency 1 --uri ${DATA_ROOT}/lancedb/lancedb-hnsw-openai-500k --m 16 --ef-construction 200 --ef 50 > lancedb-hnsw-openai-500k.txt 2>&1 &
nohup vectordbbench lancedbhnsw --case-type Performance1536D5M --k 10 --num-concurrency 1 --uri ${DATA_ROOT}/lancedb/lancedb-hnsw-openai-5m --m 16 --ef-construction 200 --ef 50 > lancedb-hnsw-openai-5m.txt 2>&1 &
```

## Artifact Evaluation

Notes for artifacte evaluation:

* We use a dedicated NVMe SSD for the storage of the database to avoid the effect of the operating system and other workloads.
* All the experiments are run in a server with a single NUMA node, we suggest pinning all the benchmark processes and vector databases in a single NUMA node to get stable results.
* The absolute performance of the vector databases may vary with different CPU and NVMe SSD configurations, we expect the relative performance of the vector databases to be stable across different configurations.
* Higher I/O bandwidth is expected if CPUs with higher performance is used.

Before run the scripts:

* The benchmarking scripts do not automatically start the docker containers, please start the docker containers before running the scripts.
* We suggest stop all the other containsers to avoid the interference of background operations of the vector databases from other docker containers.
* Please run the experiments one by one, the experiments are run in background with nohup to avoid being interrupted by unstable ssh connections.
* Make sure that the four environment variables (`PYTHON_BIN`, `VECTORDB_BENCH_BIN`, `BPFTRACE_BIN`, and `DATA_ROOT`) are set correctly before running the scripts.

The plots will appear in the `figures` directory after each experiment are finished. When all the experiments are finished, run the plot scripts to generate the final aggregated plots that appers in the paper.

### Figure 2, 3, 4

Run the experiments:

```bash
cd figure-2-3-4
# Run Milvus with IVF
nohup python3 milvus-ivf-perf.py --case-type cohere-1m --run > milvus-performance-ivf-cohere-1m.log 2>&1 &
nohup python3 milvus-ivf-perf.py --case-type cohere-10m --run > milvus-performance-ivf-cohere-10m.log 2>&1 &
nohup python3 milvus-ivf-perf.py --case-type openai-500k --run > milvus-performance-ivf-openai-500k.log 2>&1 &
nohup python3 milvus-ivf-perf.py --case-type openai-5m --run > milvus-performance-ivf-openai-5m.log 2>&1 &
# Run Milvus with HNSW
nohup python3 milvus-hnsw-perf.py --case-type cohere-1m --run > milvus-performance-hnsw-cohere-1m.log 2>&1 &
nohup python3 milvus-hnsw-perf.py --case-type cohere-10m --run > milvus-performance-hnsw-cohere-10m.log 2>&1 &
nohup python3 milvus-hnsw-perf.py --case-type openai-500k --run > milvus-performance-hnsw-openai-500k.log 2>&1 &
nohup python3 milvus-hnsw-perf.py --case-type openai-5m --run > milvus-performance-hnsw-openai-5m.log 2>&1 &
# Run Milvus with DiskANN
nohup python3 milvus-diskann-perf.py --case-type cohere-1m --run > milvus-performance-diskann-cohere-1m.log 2>&1 &
nohup python3 milvus-diskann-perf.py --case-type cohere-10m --run > milvus-performance-diskann-cohere-10m.log 2>&1 &
nohup python3 milvus-diskann-perf.py --case-type openai-500k --run > milvus-performance-diskann-openai-500k.log 2>&1 &
nohup python3 milvus-diskann-perf.py --case-type openai-5m --run > milvus-performance-diskann-openai-5m.log 2>&1 & 
# Run Qdrant with HNSW
nohup python3 qdrant-hnsw-perf.py --case-type cohere-1m --index-location memory --run > qdrant-performance-hnsw-mem-cohere-1m.log 2>&1 &
nohup python3 qdrant-hnsw-perf.py --case-type cohere-10m --index-location memory --run > qdrant-performance-hnsw-mem-cohere-10m.log 2>&1 &
nohup python3 qdrant-hnsw-perf.py --case-type openai-500k --index-location memory --run > qdrant-performance-hnsw-mem-openai-500k.log 2>&1 &
nohup python3 qdrant-hnsw-perf.py --case-type openai-5m --index-location memory --run > qdrant-performance-hnsw-mem-openai-5m.log 2>&1 &
# Run Weaviate with HNSW
nohup python3 weaviate-hnsw-perf.py --case-type cohere-1m --run > weaviate-performance-hnsw-cohere-1m.log 2>&1 &
nohup python3 weaviate-hnsw-perf.py --case-type cohere-10m --run > weaviate-performance-hnsw-cohere-10m.log 2>&1 &
nohup python3 weaviate-hnsw-perf.py --case-type openai-500k --run > weaviate-performance-hnsw-openai-500k.log 2>&1 &
nohup python3 weaviate-hnsw-perf.py --case-type openai-5m --run > weaviate-performance-hnsw-openai-5m.log 2>&1 &
# Run LanceDB with IVFPQ
nohup python3 lancedb-ivfpq-perf.py --case-type cohere-1m --run > lancedb-performance-ivfpq-cohere-1m-embedded.log 2>&1 &
nohup python3 lancedb-ivfpq-perf.py --case-type cohere-10m --run > lancedb-performance-ivfpq-cohere-10m-embedded.log 2>&1 &
nohup python3 lancedb-ivfpq-perf.py --case-type openai-500k --run > lancedb-performance-ivfpq-openai-500k-embedded.log 2>&1 &
nohup python3 lancedb-ivfpq-perf.py --case-type openai-5m --run > lancedb-performance-ivfpq-openai-5m-embedded.log 2>&1 &
# Run LanceDB with HNSW
nohup python3 lancedb-hnsw-perf.py --case-type cohere-1m --run > lancedb-performance-hnsw-cohere-1m-embedded.log 2>&1 &
nohup python3 lancedb-hnsw-perf.py --case-type cohere-10m --run > lancedb-performance-hnsw-cohere-10m-embedded.log 2>&1 &
nohup python3 lancedb-hnsw-perf.py --case-type openai-500k --run > lancedb-performance-hnsw-openai-500k-embedded.log 2>&1 &
nohup python3 lancedb-hnsw-perf.py --case-type openai-5m --run > lancedb-performance-hnsw-openai-5m-embedded.log 2>&1 &
```

After all the experiments are finished, the figures can be plotted by:

```bash
python3 plot-performance-all-dbs.py
python3 plot-cpu-all-db.py
```

### Figure 5 and 6

Run the experiments, sudo access is needed to run BPF trace.

Check the major and minor of the SSDs for traces:

```bash
ls -l PATH_TO_YOUR_SSD # Example: ls -l /dev/nvme0n1
# Example output, the major and minor are 259, 0
brw-rw---- 1 root disk 259, 0 Jun 23 03:00 /dev/nvme0n1
```

Then change the major and minor in bpfscript: ./figure-5-6/bpf-scripts/bio-trace.bt. The major and minor set in line 13.

```bash
cd figure-5-6
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type cohere-1m --concurrency 1 --run > milvus-diskann-iotrace-cohere-1m-con-1.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type cohere-1m --concurrency 32 --run > milvus-diskann-iotrace-cohere-1m-con-32.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type cohere-1m --concurrency 256 --run > milvus-diskann-iotrace-cohere-1m-con-256.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type cohere-10m --concurrency 1 --run > milvus-diskann-iotrace-cohere-10m-con-1.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type cohere-10m --concurrency 8 --run > milvus-diskann-iotrace-cohere-10m-con-8.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type cohere-10m --concurrency 256 --run > milvus-diskann-iotrace-cohere-10m-con-256.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type openai-500k --concurrency 1 --run > milvus-diskann-iotrace-openai-500k-con-1.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type openai-500k --concurrency 16 --run > milvus-diskann-iotrace-openai-500k-con-16.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type openai-500k --concurrency 256 --run > milvus-diskann-iotrace-openai-500k-con-256.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type openai-5m --concurrency 1 --run > milvus-diskann-iotrace-openai-5m-con-1.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type openai-5m --concurrency 4 --run > milvus-diskann-iotrace-openai-5m-con-4.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-iotrace.py --case-type openai-5m --concurrency 256 --run > milvus-diskann-iotrace-openai-5m-con-256.log 2>&1 &
```

After all the experiments are finished, plot the figures:

```bash
python3 plot-milvus-diskann-iotrace.py
```

### Figure 7, 8, 9, 10 and 11

Run the experiments for performance:

```bash
nohup python3 milvus-diskann-var-klist.py --case-type cohere-1m --concurrency 1 --run > milvus-var-klist-diskann-cohere-1m-concurrency-1.log 2>&1 &
nohup python3 milvus-diskann-var-klist.py --case-type cohere-10m --concurrency 1 --run > milvus-var-klist-diskann-cohere-10m-concurrency-1.log 2>&1 &
nohup python3 milvus-diskann-var-klist.py --case-type openai-500k --concurrency 1 --run > milvus-var-klist-diskann-openai-500k-concurrency-1.log 2>&1 &
nohup python3 milvus-diskann-var-klist.py --case-type openai-5m --concurrency 1 --run > milvus-var-klist-diskann-openai-5m-concurrency-1.log 2>&1 & 
```

Run the experiments for I/O traces:

```bash
sudo -E $PYTHON_BIN milvus-diskann-var-klist-io-trace.py --case-type cohere-1m --concurrency 1 --run > milvus-var-klist-diskann-cohere-1m-concurrency-1-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-klist-io-trace.py --case-type cohere-1m --concurrency 256 --run > milvus-var-klist-diskann-cohere-1m-concurrency-256-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-klist-io-trace.py --case-type cohere-10m --concurrency 1 --run > milvus-var-klist-diskann-cohere-10m-concurrency-1-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-klist-io-trace.py --case-type cohere-10m --concurrency 256 --run > milvus-var-klist-diskann-cohere-10m-concurrency-256-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-klist-io-trace.py --case-type openai-500k --concurrency 1 --run > milvus-var-klist-diskann-openai-500k-concurrency-1-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-klist-io-trace.py --case-type openai-500k --concurrency 256 --run > milvus-var-klist-diskann-openai-500k-concurrency-256-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-klist-io-trace.py --case-type openai-5m --concurrency 1 --run > milvus-var-klist-diskann-openai-5m-concurrency-1-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-klist-io-trace.py --case-type openai-5m --concurrency 256 --run > milvus-var-klist-diskann-openai-5m-concurrency-256-io-trace.log 2>&1 &
```

### Figure 12, 13, 14, 15 and 16

Setup the user config file:

```bash
cp milvus-configs/user.yaml ${DATA_ROOT}/milvus/milvus-diskann-cohere-1m/user.yaml
cp milvus-configs/user.yaml ${DATA_ROOT}/milvus/milvus-diskann-cohere-10m/user.yaml
cp milvus-configs/user.yaml ${DATA_ROOT}/milvus/milvus-diskann-openai-500k/user.yaml
cp milvus-configs/user.yaml ${DATA_ROOT}/milvus/milvus-diskann-openai-5m/user.yaml
```

Run the experiments for the performance:

```bash
nohup python3 milvus-diskann-var-bwidth.py --case-type cohere-1m --concurrency 1 --run > milvus-var-bwidth-diskann-cohere-1m-concurrency-1.log 2>&1 &
nohup python3 milvus-diskann-var-bwidth.py --case-type cohere-1m --concurrency 256 --run > milvus-var-bwidth-diskann-cohere-1m-concurrency-256.log 2>&1 &
nohup python3 milvus-diskann-var-bwidth.py --case-type cohere-10m --concurrency 1 --run > milvus-var-bwidth-diskann-cohere-10m-concurrency-1.log 2>&1 &
nohup python3 milvus-diskann-var-bwidth.py --case-type cohere-10m --concurrency 256 --run > milvus-var-bwidth-diskann-cohere-10m-concurrency-256.log 2>&1 &
nohup python3 milvus-diskann-var-bwidth.py --case-type openai-500k --concurrency 1 --run > milvus-var-bwidth-diskann-openai-500k-concurrency-1.log 2>&1 &
nohup python3 milvus-diskann-var-bwidth.py --case-type openai-500k --concurrency 256 --run > milvus-var-bwidth-diskann-openai-500k-concurrency-256.log 2>&1 &
nohup python3 milvus-diskann-var-bwidth.py --case-type openai-5m --concurrency 1 --run > milvus -var-bwidth-diskann-openai-5m-concurrency-1.log 2>&1 &
nohup python3 milvus-diskann-var-bwidth.py --case-type openai-5m --concurrency 256 --run > milvus-var-bwidth-diskann-openai-5m-concurrency-256.log 2>&1 &
```

Run the experiments for I/O traces:

```bash
sudo -E $PYTHON_BIN milvus-diskann-var-bwidth-io-trace.py --case-type cohere-1m --concurrency 1 --run > milvus-var-bwidth-diskann-cohere-1m-concurrency-1-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-bwidth-io-trace.py --case-type cohere-1m --concurrency 256 --run > milvus-var-bwidth-diskann-cohere-1m-concurrency-256-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-bwidth-io-trace.py --case-type cohere-10m --concurrency 1 --run > milvus-var-bwidth-diskann-cohere-10m-concurrency-1-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-bwidth-io-trace.py --case-type cohere-10m --concurrency 256 --run > milvus-var-bwidth-diskann-cohere-10m-concurrency-256-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-bwidth-io-trace.py --case-type openai-500k --concurrency 1 --run > milvus-var-bwidth-diskann-openai-500k-concurrency-1-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-bwidth-io-trace.py --case-type openai-500k --concurrency 256 --run > milvus-var-bwidth-diskann-openai-500k-concurrency-256-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-bwidth-io-trace.py --case-type openai-5m --concurrency 1 --run > milvus-var-bwidth-diskann-openai-5m-concurrency-1-io-trace.log 2>&1 &
sudo -E $PYTHON_BIN milvus-diskann-var-bwidth-io-trace.py --case-type openai-5m --concurrency 256 --run > milvus-var-bwidth-diskann-openai-5m-concurrency-256-io-trace.log 2>&1 &
```

### Plot with existing traces

We also provide all the traces, which can be used to plot the results with the existing traces. All the traces from the papers are under results directory.

Note: Due to the file size limitation, the I/O traces are not included in this github repository. The I/O traces can be downloaded from the Zenodo: [https://doi.org/10.5281/zenodo.16916496](https://doi.org/10.5281/zenodo.16916496).

Plot the figures with existing traces:

```bash
cd results

# Plot figures 2, 3, and 4, the plotted figures will be saved in the ./figures subdirectory
cd figure-2-3-4
python3 plot-performance-all-dbs.py
python3 plot-cpu-all-db.py

# Plot figures 5 and 6, the plotted figures will be saved in the ./figures subdirectory
cd ../figure-5-6
python3 plot-milvus-diskann-iotrace.py

# Plot figures 7, 8, 9, 10, and 11, the plotted figures will be saved in the ./figures subdirectory
cd ../figure-7-8-9-10-11
python3 plot-diskann-milvus-klist-all.py
python3 plot-diskann-milvus-klist-bandwidth.py

# Plot figures 12, 13, 14, 15, and 16, the plotted figures will be saved in the ./figures subdirectory
cd ../figure-12-13-14-15-16
python3 plot-diskann-milvus-bwidth-all.py
python3 plot-diskann-milvus-bwidth-io-trace.py
```
