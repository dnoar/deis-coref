# deis-coref
Coreference resolution project
Daniel Noar, Nicholas Miller, Evan MacPhaul

***Summary:***

We ultimately implemented a variation of the rule-based 'sieve' approach.

***Files:***

The operational scripts are the following:

- preprocess.py: tool for converting the CONLL files for use with the sieve
- main.py: tool for applying the sieve modules

Component scripts include the following:

- appositions.py: tool for finding apposition in feature file
- pronfo.py: pronoun information
- sieve_modules.py: sieve modules


Files identified as 'not implemented' are approaches that we tried, but abandoned. We include those files here for completeness:
- classifiers.py (not implemented): tools for training and evaluating various CRF classifiers 
- hobbs2.py (not implemented): tool for applying Hobb's method
- npfeats.py (not implemented): tool for finding NP features

***Instructions for running:***

Note that since the sieve is a rule-based approach, we only need to work with the test data.

1. In preprocess.py, update the CONLL_TEST constant to correspond to the directories of the corresponding CONLL files. 

2. Run preprocess.py. This will write the coref.feat file (essentially a .csv), and pickle the coreference chain data structure for use with the next steps.

3. In main.py, ensure that FEAT_PICKLE directory corresponds to the coref.pickle file created in the previous step, and that FEAT_FILE directory corresponds to the core.feat file created in step 2.

4. Run main.py. This will run the sieve on the testing data. Pandas may issue a SettingWithCopy warning, but the script will still run. The process will be slow, but you can check the progress by looking for the new files created in the new_results subfolder. 

5. Run the evaluator script on the files created by main.py, according to the evaluator script instructions.