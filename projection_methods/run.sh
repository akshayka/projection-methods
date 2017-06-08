#!/bin/bash -x

ROOT_DIR="/home/vagrant/projects/cvx_research/projection_methods/projection_methods"

ITS=300
PROBLEM_TYPE="random_socp"
PROBLEM_NAME="m_2000_n_1000"
PROBLEM_DIR="${ROOT_DIR}/problems/sv/${PROBLEM_TYPE}"
PROBLEM="${PROBLEM_DIR}/${PROBLEM_NAME}.pkl"

OUT_DIR="${ROOT_DIR}/results/${PROBLEM_TYPE}/${PROBLEM_NAME}"

python experiment.py ${PROBLEM} ${OUT_DIR}/altp altp -i $ITS -a -v
python experiment.py ${PROBLEM} ${OUT_DIR}/dyk dyk -i $ITS -a -v
python experiment.py ${PROBLEM} ${OUT_DIR}/apop apop -o exact -n apop -i $ITS -a -v
python experiment.py ${PROBLEM} ${OUT_DIR}/apop_alt apop -o exact -alt -n apop_alt -i $ITS -a -v
python results/plot_residuals.py "${OUT_DIR}/*.pkl" -o ${OUT_DIR}/plot

cp ${ROOT_DIR}/run.sh ${OUT_DIR}/run.sh
ln -s ${PROBLEM} ${OUT_DIR}/${PROBLEM_NAME}.pkl.pb
