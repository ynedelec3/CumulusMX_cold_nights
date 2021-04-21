# CumulusMX_cold_nights
 Python code for creation of graphs with cold nights temperatures from CumulusMX logs
 Python files are in src
 Suivi_temp creates and saves graphs periodically. Temperature and cumulated negative temperatures are plotted between 6 pm and 10 am.
 Only nights with temperature falling down to a minimum between two thresholds and current night are selected.
 Suivi_temp.py colors follow day chronology.
 Suivi_temp_class.py colors are set by a Time Series Kmeans classification (python module tslearn).

