CHROMOSOMES=['chr1']
DATASETS=['full']

rule all:
  input:
    # "data/basset_tfr/full/targets.txt" ,
    "data/basset_tfr/full_downsampled5p/targets.txt",
    # expand("data/basset_tfr/{dataset}_downsampled04p/targets.txt", dataset=DATASETS),
    expand("data/basset_tfr/keep_{chr}_downsampled5p/targets.txt", chr=CHROMOSOMES)
    # expand("data/basset_tfr/wo_{chr}_downsampled5p/targets.txt", chr=CHROMOSOMES),
    # expand("data/basset_model/{dataset}_downsampled5p/model_best.h5", dataset=DATASETS), # if expand option will not be used, the error prompts
    #expand('data/basset_model/{dataset}/model_best.h5', dataset=DATASETS),
    # "data/basset_tfr/full_downsampled04p/test_sequences.bed",
    # expand("results/basset_model/{dataset}_downsampled5p/acc.txt", dataset=DATASETS),
    # expand("results/basset_model/{dataset}/acc.txt", dataset=DATASETS),
    # expand("results/basset_model//{dataset}_downsampled04p/acc.txt", dataset=DATASETS),
    # "results/basset_model/model_full/test_sequences_full_downsampled04p/acc.txt",
    # "data/basset_predict/{predict_dataset}_model_{model_dataset}.bed"
    # "data/basset_predict/test_sequences_wo_chr1_downsampled5p/model_wo_chr1_downsampled5p/predict.h5"
    # "data/basset_predict/test_sequences_full/model_full/predict.h5"
    # "data/basset_predict/test_sequences_full_downsampled1p/model_full_downsampled1p/predict.h5"
    #"data/basset_predict/test_sequences_full_downsampled04p/model_full/predict.h5",
    #"results/basset_model/model_full/test_sequences_full_downsampled04p/metrics.tsv",
    #"data/basset_tfr/full_downsampled04p/hic_sequences.bed",
    # "data/basset_predict/hic_sequences_full_downsampled04p/model_full/predict.h5",
    # "data/basset_predict/hic_sequences_full_downsampled04p/model_full/avg_predict.h5", 
    # "results/basset_model/model_full/test_sequences_full_downsampled04p/hic_only_metrics.tsv",
    # "results/basset_model/model_full/test_sequences_full_downsampled04p/targets_only_metrics.tsv",
    # "results/basset_model/model_full/test_sequences_full_downsampled04p/wo_hic_metrics.tsv"

rule download_dnase:
  input:
    "config/targets.txt"
  output:
    "data/basset_like/full/targets.txt"
  shell:
    """
    mkdir data/basset_like/full/encode 
    wget -r ftp://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgDnaseUniform 
    mv hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgDnaseUniform/* data/basset_like/full/encode 
    rm -r hgdownload.cse.ucsc.edu 

    mkdir data/basset_like/full/roadmap
    wget -r -A "*DNase.hotspot.fdr0.01.peaks.bed.gz" http://egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/narrowPeak 
    mv egg2.wustl.edu/roadmap/data/byFileType/peaks/consolidated/narrowPeak/archive_DNase_hotspot_peaks/* data/basset_like/full/roadmap 
    rm -r egg2.wustl.edu 
    
    sed -e "s|data/basenji_peak/|data/basset_like/full/|" < {input} > {output}
    """

rule download_seqs:
  output:
    fasta="data/basset_like/hg19.ml.fa",
    blacklist="data/basset_like/wgEncodeHg19ConsensusSignalArtifactRegions.bed"
 
  conda:
   "../../env/samtools.yaml"
 
  shell:
    """

    curl -o {output.fasta} https://storage.googleapis.com/basenji_tutorial_data/hg19.ml.fa
    samtools faidx {output.fasta}
   
    curl -O https://personal.broadinstitute.org/anshul/projects/encode/rawdata/blacklists/wgEncodeHg19ConsensusSignalArtifactRegions.bed.gz
    gunzip wgEncodeHg19ConsensusSignalArtifactRegions.bed.gz
    mv wgEncodeHg19ConsensusSignalArtifactRegions.bed {output.blacklist}
    
    """

rule leave_out_one_chr:
  input:
    "data/basset_like/full/targets.txt"
  output:
    "data/basset_like/wo_{chr}/targets.txt"

  shell:
    """
    mkdir -p data/basset_like/wo_{wildcards.chr}/encode/
    mkdir -p data/basset_like/wo_{wildcards.chr}/roadmap/
    
    for f in $(cd data/basset_like/full/encode; ls *.narrowPeak.gz); do
      zcat data/basset_like/full/encode/$f |
        grep -v -P "^{wildcards.chr}\t" |
        gzip > data/basset_like/wo_{wildcards.chr}/encode/$f
    done

    for f in $(cd data/basset_like/full/roadmap; ls *.bed.gz); do
      zcat data/basset_like/full/roadmap/$f |
        grep -v -P "^{wildcards.chr}\t" |
        gzip > data/basset_like/wo_{wildcards.chr}/roadmap/$f
    done

    sed -e "s|data/basset_like/full/|data/basset_like/wo_{wildcards.chr}/|" < {input} > {output}
    """
### to have only with chr 1 fix "-v" argument

rule keep_chr:
  input:
    "data/basset_like/full/targets.txt"
  output:
    "data/basset_like/keep_{chr}/targets.txt"

  shell:
    """
    mkdir -p data/basset_like/keep_{wildcards.chr}/encode/
    mkdir -p data/basset_like/keep_{wildcards.chr}/roadmap/
    
    for f in $(cd data/basset_like/full/encode; ls *.narrowPeak.gz); do
      zcat data/basset_like/full/encode/$f |
        grep -P "^{wildcards.chr}\t" |
        gzip > data/basset_like/keep_{wildcards.chr}/encode/$f
    done

    for f in $(cd data/basset_like/full/roadmap; ls *.bed.gz); do
      zcat data/basset_like/full/roadmap/$f |
        grep -P "^{wildcards.chr}\t" |
        gzip > data/basset_like/keep_{wildcards.chr}/roadmap/$f
    done

    sed -e "s|data/basset_like/full/|data/basset_like/keep_{wildcards.chr}/|" < {input} > {output}
    """

rule make_data:
  input:
    fasta="data/basset_like/hg19.ml.fa",
    blacklist="data/basset_like/wgEncodeHg19ConsensusSignalArtifactRegions.bed",
    targets="data/basset_like/{dataset}/targets.txt"

  output:
    "data/basset_tfr/{dataset}/targets.txt", 
    "data/basset_tfr/{dataset}/sequences.bed"


  conda:
    "basenji-gpu"

  shell:
    """
    rmdir data/basset_tfr/{wildcards.dataset}

    basenji_data.py -b {input.blacklist} \
      --local -p 8 -r 4096 -w 192 -l 1344 --peaks -v .12 -t .12 --stride 192 --stride_test 192 --crop 576 \
      -o data/basset_tfr/{wildcards.dataset} {input.fasta} {input.targets}
    """

rule make_data_downsampled5p:
  input:
    fasta="data/basset_like/hg19.ml.fa",
    blacklist="data/basset_like/wgEncodeHg19ConsensusSignalArtifactRegions.bed",
    targets="data/basset_like/{dataset}/targets.txt"

  output:
    "data/basset_tfr/{dataset}_downsampled5p/targets.txt",
    "data/basset_tfr/{dataset}_downsampled5p/sequences.bed"

  conda:
    "basenji-gpu"

  shell:
    """
    rmdir data/basset_tfr/{wildcards.dataset}_downsampled5p

    basenji_data.py -b {input.blacklist} -d 0.05 \
      --local -p 8 -r 4096 -w 192 -l 1344 --peaks -v .12 -t .12 --stride 192 --stride_test 192 --crop 576 \
      -o data/basset_tfr/{wildcards.dataset}_downsampled5p {input.fasta} {input.targets}
    """

rule make_data_downsampled04p:
  input:
    fasta="data/basset_like/hg19.ml.fa",
    blacklist="data/basset_like/wgEncodeHg19ConsensusSignalArtifactRegions.bed",
    targets="data/basset_like/{dataset}/targets.txt"

  output:
    "data/basset_tfr/{dataset}_downsampled04p/targets.txt",
    "data/basset_tfr/{dataset}_downsampled04p/sequences.bed"

  conda:
    "basenji-gpu"

  shell:
    """
    rmdir data/basset_tfr/{wildcards.dataset}_downsampled04p

    basenji_data.py -b {input.blacklist} -d 0.004 \
      --local -p 8 -r 4096 -w 192 -l 1344 --peaks -v .12 -t .12 --stride 192 --stride_test 192 --crop 576 \
      -o data/basset_tfr/{wildcards.dataset}_downsampled04p {input.fasta} {input.targets}
    """

ruleorder:
  make_data_downsampled5p > make_data


rule train:
  input:
    params="config/params_basset.json"

  output:
    targets="data/basset_model/{dataset}/model_best.h5"

  threads:
    6

  conda:
    "basenji-gpu"

  shell:
    """
    export LD_LIBRARY_PATH=/home/dkokot/anaconda3/envs/basenji_gpu/lib:/usr/local/cuda/extras/CUPTI/lib64:
    export PATH=/home/dkokot/Basenji_HiC/basenji/bin:$PATH
    
    rmdir data/basset_model/{wildcards.dataset}
    
    CUDA_VISIBLE_DEVICES=0 basenji_train.py -o data/basset_model/{wildcards.dataset} \
      {input.params} data/basset_tfr/{wildcards.dataset}
    """

rule test:
  input:
    model="data/basset_model/{model_dataset}/model_best.h5",
    params="config/params_basset.json"
  output:
    result="results/basset_model/model_{model_dataset}/test_sequences_{sequences_dataset}/acc.txt"

  conda:
    "basenji_gpu"
  
  shell:
    """
    
    export PATH=/home/dkokot/Basenji_HiC/basenji/bin:$PATH

    rmdir results/basset_model/model_{wildcards.model_dataset}/test_sequences_{wildcards.sequences_dataset}
    taskset -c 20-30 basenji_test.py -o results/basset_model/model_{wildcards.model_dataset}/test_sequences_{wildcards.sequences_dataset} \
      --rc --shifts "1,0,-1" \
      --save {input.params} {input.model} data/basset_tfr/{wildcards.sequences_dataset} | tee logs/{wildcards.sequences_dataset}_test.txt
    """
    # export LD_LIBRARY_PATH=/home/dkokot/anaconda3/envs/basenji_gpu/lib:/usr/local/cuda/extras/CUPTI/lib64:

# rule make_test_sequences:
#   input:
#     "data/basset_tfr/{dataset}/sequences.bed"
#   output:  
#     "data/basset_tfr/{dataset}/test_sequences.bed"

#   shell:
#     """
#     grep "test" {input} > {output}
#     """



# rule predict:
#   input:
#     model="data/basset_model/{model_dataset}/model_best.h5", 
#     bed="data/basset_tfr/{sequences_dataset}/test_sequences.bed", 
#     params="config/params_basset.json", 
#     fasta="data/basset_like/hg19.ml.fa",

#   output:
#     "data/basset_predict/test_sequences_{sequences_dataset}/model_{model_dataset}/predict.h5"

#   conda:
#     "basenji_gpu"
  
#   shell:
#     """
    
#     export PATH=/home/dkokot/Basenji_HiC/basenji/bin:$PATH

#     time taskset -c 20-30 basenji_predict_bed.py -o data/basset_predict/test_sequences_{wildcards.sequences_dataset}/model_{wildcards.model_dataset} \
#       --rc --shifts "1,0,-1" \
#       -f {input.fasta} {input.params} {input.model} {input.bed}
#     """

# # export LD_LIBRARY_PATH=/home/dkokot/anaconda3/envs/basenji_gpu/lib:/usr/local/cuda/extras/CUPTI/lib64:
# rule predict_hic:
#   input:
#     model="data/basset_model/{model_dataset}/model_best.h5", 
#     bed="data/basset_tfr/{sequences_dataset}/hic_sequences.bed", 
#     params="config/params_basset.json", 
#     fasta="data/basset_like/hg19.ml.fa",

#   output:
#     "data/basset_predict/hic_sequences_{sequences_dataset}/model_{model_dataset}/predict.h5"

#   conda:
#     "basenji_gpu"
  
#   shell:
#     """
    
#     export PATH=/home/dkokot/Basenji_HiC/basenji/bin:$PATH

#     time taskset -c 20-30 basenji_predict_bed.py -o data/basset_predict/hic_sequences_{wildcards.sequences_dataset}/model_{wildcards.model_dataset} \
#       --rc --shifts "1,0,-1" \
#       -f {input.fasta} {input.params} {input.model} {input.bed}
#     """


# rule avg_preds:
#   input:
#     preds_hic = "data/basset_predict/hic_sequences_{sequences_dataset}/model_{model_dataset}/predict.h5", 
#     preds_tsv = "results/hic_output/{sequences_dataset}.tsv"

#   output:
#    hic_only = "data/basset_predict/hic_sequences_{sequences_dataset}/model_{model_dataset}/hic_only_predict.h5", 
#    avg = "data/basset_predict/hic_sequences_{sequences_dataset}/model_{model_dataset}/avg_predict.h5"

#   conda:
#     "basenji_gpu"

#   shell:
#     """
#     python src/py/avg_hic_pred.py -p {input.preds_hic} --tsv {input.preds_tsv} -o {output.avg} \
#     python src/py/avg_hic_pred.py -p {input.preds_hic} --tsv {input.preds_tsv} -o {output.hic_only}
#     """



# rule metrics:
#   input:
#     predictions="data/basset_predict/test_sequences_{sequences_dataset}/model_{model_dataset}/predict.h5",
#     targets="results/basset_model/model_{model_dataset}/test_sequences_{sequences_dataset}/targets.h5"
  
#   output:
#     metrics="results/basset_model/model_{model_dataset}/test_sequences_{sequences_dataset}/metrics.tsv"

#   conda:
#     "basenji_gpu"

#   shell:
#     """
#     python src/py/metrics.py {input.predictions} {input.targets} {output.metrics}

#     """

# rule metrics_hic_only:
#   input:
#     predictions="data/basset_predict/hic_sequences_{sequences_dataset}/model_{model_dataset}/hic_only_predict.h5",
#     targets = "results/basset_model/model_{model_dataset}/test_sequences_{sequences_dataset}/targets.h5"

#   output:
#     only_hic = "results/basset_model/model_{model_dataset}/test_sequences_{sequences_dataset}/hic_only_metrics.tsv"

#   conda:
#     "basenji_gpu"

#   shell:
#     """
#     python src/py/metrics.py {input.predictions} {input.targets} {output.metrics}

#     """

# rule metrics_wo_hic:
#   input:
#     predictions="data/basset_predict/hic_sequences_{sequences_dataset}/model_{model_dataset}/avg_predict.h5",
#     targets="results/basset_model/model_{model_dataset}/test_sequences_{sequences_dataset}/targets.h5"
  
#   output:
#     metrics="results/basset_model/model_{model_dataset}/test_sequences_{sequences_dataset}/wo_hic_metrics.tsv"

#   conda:
#     "basenji_gpu"

#   shell:
#     """
#     python src/py/metrics.py {input.predictions} {input.targets} {output.metrics}

#     """

# rule extract_hic:
#   input:
#     hic="data/hic/GSE63525_K562_combined_30.hic",
#     targets="data/basset_tfr/{test_sequences}/test_sequences.bed"

#   output:
#     tsv="results/hic_output/{test_sequences}_hic.tsv",
#     bed="data/basset_tfr/{test_sequences}/hic_sequences.bed"
#   conda:
#     "hic-straw"

#   shell:
#     """
#     taskset -c 20-30 python src/py/extract_hic.py {input.hic} \
#     --targets {input.targets} \
#     --resolution 5000 --range 1000000 \
#     --tsv {output.tsv} \
#     --bed {output.bed}
#     """
  