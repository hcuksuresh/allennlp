"""
The ``predict`` subcommand allows you to make bulk JSON-to-JSON
predictions using a trained model and its :class:`~allennlp.service.predictors.predictor.Predictor` wrapper.

.. code-block:: bash

    $ allennlp predict --help
    usage: allennlp [command] predict [-h]
                                      [--output-file OUTPUT_FILE]
                                      [--batch-size BATCH_SIZE]
                                      [--silent]
                                      [--cuda-device CUDA_DEVICE]
                                      [-o OVERRIDES]
                                      [--include-package INCLUDE_PACKAGE]
                                      [--predictor PREDICTOR]
                                      archive_file input_file

    Run the specified model against a JSON-lines input file.

    positional arguments:
    archive_file          the archived model to make predictions with
    input_file            path to input file

    optional arguments:
    -h, --help            show this help message and exit
    --output-file OUTPUT_FILE
                            path to output file
    --batch-size BATCH_SIZE
                            The batch size to use for processing
    --silent              do not print output to stdout
    --cuda-device CUDA_DEVICE
                            id of GPU to use (if any)
    -o OVERRIDES, --overrides OVERRIDES
                            a HOCON structure used to override the experiment
                            configuration
    --include-package INCLUDE_PACKAGE
                            additional packages to include
    --predictor PREDICTOR
                            optionally specify a specific predictor to use
"""

import argparse
from contextlib import ExitStack
import sys
from typing import Optional, IO
from datetime import datetime

from allennlp.commands.subcommand import Subcommand
from allennlp.models.archival import load_archive
from allennlp.service.predictors import Predictor
import json
import PyPDF2
import dill
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


class Predict(Subcommand):
     #reating a pdf file object
    #print(datetime.now(),"  --  Start File Reading")
    #pdfFileObj = open('EmployeeHandbook_January_2018.pdf', 'rb')
     #pdfFileObj = open(origFileName, 'rb')
  
     # creating a pdf Reader object
    #pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
  
     # creating a pdf writer object for new pdf
    #pdfWriter = PyPDF2.PdfFileWriter()
    #emp_hb=[]
     # rotating each page
    #for page in range(pdfReader.numPages): 
     # pageObj = pdfReader.getPage(page)
      #emp_hb.append(pageObj.extractText())
    
   # global emp_handbook
   # pdfFileObj.close()
    #emp_handbook="".join(emp_hb)
    #print(datetime.now(),"  --  Complete File Read")
    
    #def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
       
        #print ("Enter your question")
        #question = input()
        #print(datetime.now(), "  --  Time after entering Question")
        #d={}
        #d['distractor1']=""
        #d['question']=question
        #d['distractor3']=""
        #d['passage']=emp_handbook
        #d['correct_answer']=""
        #d['distractor2']=""
        #print (d)
        #with open('data_emp_hb.jsonl', 'w') as outfile:
         #   json.dump(d, outfile)

    #global filename
    #filename = 'category.pkl'
    #dill.load_session(filename)
    #cat = (clf.predict(count_vect.transform(["commitment"])))
    #import pandas as pd
    category = pd.read_csv('emp_hb_train_set.csv', header=None, encoding = 'unicode_escape')
    category.columns = ['topic', 'description']

    category = category.dropna()
    category['category_id'] = category['topic'].factorize()[0]
    category_id_df = category[['topic', 'category_id']].drop_duplicates().sort_values('category_id')
    category_to_id = dict(category_id_df.values)
    id_to_category = dict(category_id_df[['category_id', 'topic']].values)

    category.head()
    X_train, X_test, y_train, y_test = train_test_split(category['description'], category['topic'])
    global count_vect
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    global clf
    clf = MultinomialNB().fit(X_train_tfidf, y_train)


    def add_subparser(self, name: str, parser: argparse._SubParsersAction) -> argparse.ArgumentParser:
      
        print ("Enter your question")
        question = input()
        cat = (clf.predict(count_vect.transform([question])))
        #print(cat)
        category = pd.read_csv('match_cat.csv', header=None, encoding = 'unicode_escape')
        category.columns = ['topic', 'description']
        category= category.dropna()
        text_to_mrc =(category.loc[category['topic'] == cat[0]]['description'].item())
        #print(text_to_mrc)
        #hit=input("hit")
        #text_to_mrc=text_to_mrc[0]
        print (text_to_mrc)
        d={}
        d['distractor1']=""
        d['question']=question
        d['distractor3']=""
        d['passage']=text_to_mrc
        d['correct_answer']=""
        d['distractor2']=""
        #print (d)
        with open('data_emp_hb_category.jsonl', 'w') as outfile:
            json.dump(d, outfile)
        description = '''Run the specified model against a JSON-lines input file.'''
        subparser = parser.add_parser(
                name, description=description, help='Use a trained model to make predictions.')

        subparser.add_argument('archive_file', type=str, help='the archived model to make predictions with')
        subparser.add_argument('input_file', type=argparse.FileType('r'), help='path to input file')

        subparser.add_argument('--output-file', type=argparse.FileType('w'), help='path to output file')
        subparser.add_argument('--weights-file',
                               type=str,
                               help='a path that overrides which weights file to use')

        batch_size = subparser.add_mutually_exclusive_group(required=False)
        batch_size.add_argument('--batch-size', type=int, default=1, help='The batch size to use for processing')

        subparser.add_argument('--silent', action='store_true', help='do not print output to stdout')

        cuda_device = subparser.add_mutually_exclusive_group(required=False)
        cuda_device.add_argument('--cuda-device', type=int, default=-1, help='id of GPU to use (if any)')

        subparser.add_argument('-o', '--overrides',
                               type=str,
                               default="",
                               help='a HOCON structure used to override the experiment configuration')

        subparser.add_argument('--predictor',
                               type=str,
                               help='optionally specify a specific predictor to use')

        subparser.set_defaults(func=_predict)

        return subparser

def _get_predictor(args: argparse.Namespace) -> Predictor:
    archive = load_archive(args.archive_file,
                           weights_file=args.weights_file,
                           cuda_device=args.cuda_device,
                           overrides=args.overrides)
    #oprint (archive) 
    #hit =input("hit")
    return Predictor.from_archive(archive, args.predictor)

def _run(predictor: Predictor,
         input_file: IO,
         output_file: Optional[IO],
         batch_size: int,
         print_to_console: bool) -> None:

    def _run_predictor(batch_data):
        #print(batch_data)
        if len(batch_data) == 1:
            result = predictor.predict_json(batch_data[0])
            # Batch results return a list of json objects, so in
            # order to iterate over the result below we wrap this in a list.
            results = [result]
            #print(results)
        else:
            results = predictor.predict_batch_json(batch_data)

        for model_input, output in zip(batch_data, results):
                       
            string_output = predictor.dump_line(output)
            #print (type(string_output))
            if print_to_console:
                for key, value in output.items():
                  if key=='best_span_str':
                      print ("Answer: "+str(value))
                      print(datetime.now(), "  --  Time after Answer")

                #print("input: ", model_input)
                #print("prediction: ", string_output)
            if output_file:
                output_file.write(string_output)

    batch_json_data = []
    for line in input_file:
        if not line.isspace():
            # Collect batch size amount of data.
            json_data = predictor.load_line(line)
            batch_json_data.append(json_data)
            if len(batch_json_data) == batch_size:
                _run_predictor(batch_json_data)
                batch_json_data = []

    # We might not have a dataset perfectly divisible by the batch size,
    # so tidy up the scraps.
    if batch_json_data:
        _run_predictor(batch_json_data)
    #print (batch_json_data)

def _predict(args: argparse.Namespace) -> None:
    predictor = _get_predictor(args)
    output_file = None

    if args.silent and not args.output_file:
        print("--silent specified without --output-file.")
        print("Exiting early because no output will be created.")
        sys.exit(0)

    # ExitStack allows us to conditionally context-manage `output_file`, which may or may not exist
    with ExitStack() as stack:
        input_file = stack.enter_context(args.input_file)  # type: ignore
        if args.output_file:
            output_file = stack.enter_context(args.output_file)  # type: ignore

        _run(predictor,
             input_file,
             output_file,
             args.batch_size,
             not args.silent)
