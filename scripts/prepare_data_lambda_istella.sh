set -x
set -e

ALG=lambda_mart
LENGTH_CUT=40

LETOR_DATA_PATH=./data/istella_letor/split/
INIT_OUTPUT_PATH=./processed_data/lambda_mart_output_istella_1000_20_ndcg10/

OUTPUT_PATH=./processed_data/${ALG}_rank_data_istella_ndcg10_1_${LENGTH_CUT}/
mkdir ${OUTPUT_PATH}
python scripts/prepare_istella_data.py $LETOR_DATA_PATH $INIT_OUTPUT_PATH $OUTPUT_PATH ${LENGTH_CUT} 0

OUTPUT_PATH=./processed_data/${ALG}_rank_data_istella_ndcg10_1_${LENGTH_CUT}_sh/
mkdir ${OUTPUT_PATH}
python scripts/prepare_istella_data.py $LETOR_DATA_PATH $INIT_OUTPUT_PATH $OUTPUT_PATH ${LENGTH_CUT} 1
