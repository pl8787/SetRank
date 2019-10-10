set -e
set -x

TAG=lambda_mart_output_istella_1000_20_ndcg10
OUTPUT_DIR=./processed_data/${TAG}/
mkdir -p $OUTPUT_DIR

ROOT_PATH=./data/istella_letor/split/
TRAIN_FILE=$ROOT_PATH/train.txt
VALID_FILE=$ROOT_PATH/valid.txt
TEST_FILE=$ROOT_PATH/test.txt
OUTPUT_PATH=$OUTPUT_DIR/lambda_mart.model
java -jar ./ranklib/RankLib-2.12.jar -train $TRAIN_FILE -ranker 6 -validate $VALID_FILE -test $TEST_FILE -save $OUTPUT_PATH -tree 1000 -metric2t NDCG@10 -leaf 20
SCORE_PATH=$OUTPUT_DIR

java -jar ./ranklib/RankLib-2.12.jar -load $OUTPUT_PATH -rank $TRAIN_FILE -score ${SCORE_PATH}/train.predict.raw 
java -jar ./ranklib/RankLib-2.12.jar -load $OUTPUT_PATH -rank $VALID_FILE -score ${SCORE_PATH}/valid.predict.raw 
java -jar ./ranklib/RankLib-2.12.jar -load $OUTPUT_PATH -rank $TEST_FILE -score ${SCORE_PATH}/test.predict.raw 

cut -f 3 ${SCORE_PATH}/train.predict.raw > ${SCORE_PATH}/train.predict
cut -f 3 ${SCORE_PATH}/valid.predict.raw > ${SCORE_PATH}/valid.predict
cut -f 3 ${SCORE_PATH}/test.predict.raw > ${SCORE_PATH}/test.predict
