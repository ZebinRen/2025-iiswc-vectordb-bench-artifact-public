nohup /home/zebin/anaconda3/envs/vectordb-bench-new/bin/vectordbbench lancedbivfpq --skip-drop-old --skip-load --case-type Performance768D1M --k 10 --num-concurrency 1 --uri /mnt/vectordb/nvme0n1/lancedb/lancedb-ivfpq-cohere-1m --num-partitions 4000 --nprobes 25 > lancedb-ivfpq-cohere-1m-concurrency-1.log 2>&1 &

nohup /home/zebin/anaconda3/envs/vectordb-bench-new/bin/vectordbbench lancedbivfpq --skip-drop-old --skip-load --case-type Performance768D10M --k 10 --num-concurrency 1 --uri /mnt/vectordb/nvme0n1/lancedb/lancedb-ivfpq-cohere-10m --num-partitions 12648 --nprobes 17 > lancedb-ivfpq-cohere-10m-concurrency-1.log 2>&1 &

nohup /home/zebin/anaconda3/envs/vectordb-bench-new/bin/vectordbbench lancedbivfpq --skip-drop-old --skip-load --case-type Performance1536D500K --k 10 --num-concurrency 1 --uri /mnt/vectordb/nvme0n1/lancedb/lancedb-ivfpq-openai-500k --num-partitions 2828 --nprobes 16 > lancedb-ivfpq-openai-500k-concurrency-1.log 2>&1 &

nohup /home/zebin/anaconda3/envs/vectordb-bench-new/bin/vectordbbench lancedbivfpq --skip-drop-old --skip-load --case-type Performance1536D5M --k 10 --num-concurrency 1 --uri /mnt/vectordb/nvme0n1/lancedb/lancedb-ivfpq-openai-5m --num-partitions 8944 --nprobes 11 > lancedb-ivfpq-openai-5m-concurrency-1.log 2>&1 &
