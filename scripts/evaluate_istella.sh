ROOT=./
${ROOT}/galago/core/target/appassembler/bin/galago eval --judgments=${ROOT}/processed_data/lambda_mart_rank_data_istella_ndcg10_1_40/test/test.qrels \
     --baseline=$1 \
     --metrics+ndcg1 \
     --metrics+ndcg3 \
     --metrics+ndcg5 \
     --metrics+ndcg10 \
     --metrics+ERR1 \
     --metrics+ERR3 \
     --metrics+ERR5 \
     --metrics+ERR10
