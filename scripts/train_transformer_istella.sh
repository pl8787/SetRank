set -x

export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=2

ROOT_DIR=./

TRAIN_DIR=$ROOT_DIR/model/train_transformer_istella/
TEST_DIR=$ROOT_DIR/model/test_transformer_istella/
PARAMS="learning_rate=0.0005,num_blocks=6,hidden_units=256,num_heads=8,dropout_rate=0.0,empty_bound=-10.0,activation=relu,use_mask=True,l2_loss=0.0,gaussian_kernel=100,multi_abstract=True,num_induced=20,pos_embed=False,pos_embed_multi=1"

mkdir -p $TRAIN_DIR
mkdir -p $TEST_DIR

DATA_PATH=$ROOT_DIR/processed_data/lambda_mart_rank_data_istella_ndcg10_1_40_sh/

python ${ROOT_DIR}/src/main_transformer.py --data_dir $DATA_PATH --train_dir $TRAIN_DIR --test_dir $TEST_DIR --steps_per_checkpoint 100 --max_train_iteration 10000 --hparams=$PARAMS --batch_size 32

python ${ROOT_DIR}/src/main_transformer.py --data_dir $DATA_PATH --train_dir $TRAIN_DIR --test_dir $TEST_DIR/eval_ --steps_per_checkpoint 200 --max_train_iteration 10000 --decode --hparams=$PARAMS --batch_size 64 

sh ${ROOT_DIR}/scripts/evaluate_istella.sh ${TEST_DIR}/eval_test.ranklist | tee ${TEST_DIR}/result.log
