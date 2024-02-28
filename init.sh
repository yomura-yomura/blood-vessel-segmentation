#!/bin/sh

download_competition_dataset () {
  desc=$1
  dataset=$2
  dataset_fn="$(basename -- $dataset)"

  echo "* Download the $desc dataset: $dataset"
  kaggle competitions download -c $dataset && \
  unzip -o ${dataset_fn}.zip -d data/$dataset_fn
  rm -f ${dataset_fn}.zip
}

# Official Dataset

download_competition_dataset "kaggle competitions" "blood-vessel-segmentation"

#

download_dataset () {
  desc=$1
  dataset=$2
  dataset_fn="$(basename -- $dataset)"

  echo "* Download the $desc dataset: $dataset"
  kaggle datasets download -d $dataset && \
  unzip -o ${dataset_fn}.zip -d data/$dataset_fn
  rm -f ${dataset_fn}.zip
}

#download_dataset "train dataset with part_id " "jumtras1/train-series-with-partid"
#download_dataset "train train datasets" "ranchantan/cmi-dss-train-datasets"
#download_dataset "train k-fold indices" "ranchantan/cmi-dss-train-k-fold-indices"


#

download_kernel_output () {
  desc=$1
  dataset=$2
  dataset_fn="$(basename -- $dataset)"

  echo "* Download the $desc dataset: $dataset"
  kaggle kernels output $dataset -p data/$dataset_fn
}

#download_kernel_output "Sleep-Critical-point-Prepare-Data" "werus23/sleep-critical-point-prepare-data"
