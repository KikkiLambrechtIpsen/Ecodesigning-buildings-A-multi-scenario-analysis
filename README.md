# Eco-desing in the building sector

This is the code used in the analysis presented in:

_Ipsen K L, Pizzol M, Birkved M, Amor B, How differentiating design strategies across building components lead to maximum reduction of adverse environmental impacts, journal unknown (XXXX),_[doi....](https://doi.org/....).

The supplementary materials to the article include the raw data: 

- LCI tables: "Database_y1", "Database_y2", "Database_y3", "Database_y4", "Database_y5", "Database_y6", "Database_y7", and "Database_y8".
- Treated results:"optimal_solution_counted" and "worst_solution_counted"

These are needed to run the script contained in this repository, for example if you want to reproduce the results of the paper. 

To run the script in this repository the following is also needed: 

- brightway2 
- ecoinvent 3.8 consequential database for brigtway2
- numpy
- pandas
- itertools
- seaborn
- matplotlib
- statsmodels

##The repository includes:

`Eco-designing buildings_A multi-scenario analysis.py` Python script to reproduce results of the LCA using the brightway2 LCA software. The scipt imports the LCIs, performs LCA calculations, and performs the statistical analysis on the resutls.

`IW_bw2.bw2package` bw2package file to implement IMPACT World+ impact assessment method in Brightway2. Reference: http://doi.org/10.5281/zenodo.3521041`lci_to_bw2.py` Python script to import inventory tables in .csv directly into brightway2.

`Iteration file_impact assessment (part 2 in script)` text file containing some of the iterations to run in the `Eco-designing buildings_A multi-scenario analysis.py` to reproduce the results of the article.

`Iteration file_distribution (part 4 in script)` text file containing some of the iterations to run in the `Eco-designing buildings_A multi-scenario analysis.py` to reproduce the results of the article.

`Iteration file_regression (part 5 in script)`text file containing some of the iterations to run in the `Eco-designing buildings_A multi-scenario analysis.py` to reproduce the results of the article.

`Iteration file_best-worst distribution (part 6.2.1 in script)`text file containing some of the iterations to run in the `Eco-designing buildings_A multi-scenario analysis.py` to reproduce the results of the article.

`Iteration file_optimize one impact category (part 7 in script)`text file containing some of the iterations to run in the `Eco-designing buildings_A multi-scenario analysis.py` to reproduce the results of the article.