# GROUP NUMBER:	21

# STUDENTS:
  -  Francesco Lazzari  1917922
  -  Riccardo  Violano  2148833


# NOTE:
To perform the work on cassandra we needed to run it on a docker container, there are all the instructions inside the `main.ipynb` file.

# MATERIALS:
  -  'main.ipynb'                  -> In this file there are all the code that we used to manage the data, create the aggregations, and also create the queries and execute them.
  -  'Data'                        -> Folder with the datasets created after the preprocessing for HW2 
                                      these tables contain the records of all the table created for the SQL refined schema for HW2 (NOTE: HW1 & HW2 have different schemas) 
				      these files can be obtained running the preprocessing attached into the submission for HW1 & HW2 but for simplicity we have add them here

# ADDITIONAL MATERIALS:
In order to check that all pur queries are the same (or almost the same [see `main`]) we have attached also the required files for the SQL database creation, record insert and query execution 
  -  'HW2_Database_creation.sql'  -> SQL code for the creation of the refined schema for HW2
  -  'HW2_query.sql'              -> SQL code containing the optimized query for the HW2 assignment
  -  '⁠hw2_global_insert'          -> SQL file to insert all the record for the schemas of HW2, but it is a bit large, if there are some problems with this file 
                                     there is also the possibility to create some smaller insert in the `Final Preprocessing.ipynb` attached into the previous submission 
				     (NOTE: the `Final Preprocessing.ipynb` for HW1 & HW2 needs the original Data that are also attached into the previous submission)