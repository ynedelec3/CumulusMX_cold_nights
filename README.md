# CumulusMX_cold_nights
 Python code for creation of graphs with cold nights temperatures from CumulusMX logs
 Python files are in src
 Suivi_temp creates and saves graphs periodically during a fixed-length cycle once started. Temperature and cumulated negative temperatures are plotted (2 plots, updated) between 6 pm and 10 am.
 Only nights with temperature falling down to a minimum between two thresholds and current night are shown.
 Suivi_temp.py colors follow day chronology.
 Suivi_temp_class.py is derived from Suivi_temp.py, additional graphs with night classification (Time Series Kmeans classification, python module tslearn) are drawn.
 Classification is based upon temperature time series and cumulated temperatures time series (4 plots).
 Suivi_temp_class.py colors are set according to night class.


