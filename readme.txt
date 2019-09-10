-- The app is stored within virtual environment <sentiance_vb>.
-- For perform project file do not forget import custom module file from 
   <custom_modules_vb\vb_utils.py>.
-- An extra column <Person> to the dataframe of a person is appended to 
   index each person while analyzing.
-- Q3 Explanation: Idea is to perform following steps:
       #1. Cluster locations for each person.
       #2. Check how many travels occured in each class for specified person.
           If number of travels is less than defined threshold, remove that 
           class assuming that this class in meaningless, random, or incorrect.
       #3. Zoom in to each class per user subset to detect where the user
           starts day's travels (<HOME>), where stays major part of day (<WORK>) 
           and where come backat evenings (<HOME>).