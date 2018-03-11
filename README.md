# deis-coref
Coreference resolution project
Daniel Noar, Nicholas Miller, Evan MacPhaul

***Files:***

- appositions.py: tool for finding apposition in feature file
- classifiers.py: tools for training and evaluating classifier
- hobbs2.py: tool for applying Hobb's method (not implemented)
- npfeats.py: tool for finding NP features (not implemented)


- preprocess.py: tool for converting the CONLL files for use with the sieve
- main.py: tool for applying the sieve modules
- pronfo.py: pronoun information
- sieve_modules.py: sieve modules

***Instructions for running:***

Note that since the sieve is a rule-based approach, we only need to work with the test data.

1. In preprocess.py, update the CONLL_TRAIN, CONLL_DEV, and CONLL_TEST constants to point to the directories of the corresponding CONLL files.

2. Run preprocess.py. This will write the coref.feat file (essentially a .csv), and pickle the coreference chain data structure for use with the next steps.

3. In main.py, ensure that FEAT_PICKLE directory corresponds to the coref.pickle file created in the previous step, and that FEAT_FILE directory corresponds to the core.feat file created in step 2.

4. Run main.py. This will run the sieve on the testing data. Pandas may issue a SettingWithCopy warning, but the script will still run. The process will be slow, but you can check the progress by looking for the new files created in the new_results subfolder. 

5. Run the evaluator script on the files created by main.py, according to the evaluator script instructions.