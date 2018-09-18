python -m allennlp.run predict bidaf-model-2017.09.15-charpad.tar.gz data_emp_hb_category.jsonl

### Added ML algo to predict user question category, accodingly passage will be passed to MRC.
*../allennlp/allennlp/commands/predict.py*
