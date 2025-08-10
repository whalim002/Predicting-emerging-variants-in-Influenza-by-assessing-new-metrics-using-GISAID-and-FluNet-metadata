INSTRUCTIONS:
    1. Run 'trial_documented_revised_all_country_script.py' to clean, process, and acquire the ratio of #FluNet_Cases/#GISAID_Cases in the 'analysis_script' folder
	--- Directory in newannotator: /HOME_ann/BII/biihalimwn/analysis_script/trial_documented_revised_all_country_script.py
    2. Run 'weighted_location_plotter_smoothed.py' to plot the line graphs (classified by each subtype) to assess the fluctuations/peaks/trends of each country throughout the specified periods and time-window. The 'plots' folder is created within the output directory from script (1).

'trial_documented_revised_all_country_script.py'
    - Automatically import the 'flunet_file' from WHO (FluNet) metadata directly from the website: "https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv"
        -> 'VIW_FNT.csv' will be acquired and renamed as 'flunet_metadata.tsv' - saved in the input directory
    - The GISAID file will be imported directly from 'accession2metadata.tsv' (updated as of 2025-08)
    - 'output_dir' is adjusted according to your preferred output directory
    - AIM: An automatic pipeline to analyze the #FluNet_Cases/#GISAID_Samples, merging the FluNet's and pre-existing GISAID's databases.
    - OUTPUT: The output directory contains separate files for every subtypes, with emphasis on AH1N12009.tsv, AH3.tsv, and BVIC.tsv.
    - MAPPING RESOLUTION: - Successfully mapped 174 countries (sourced from gisaid_df['currCountry'] and flunet_df['COUNTRY_AREA_TERRITORY']). All countries (with >= 70% words' matching are saved in 'high_accuracy_country_mappings.tsv').
                          - Low accuracy and unmapped countries are saved respectively in 'low_accuracy_country_mappings.tsv' and 'unmapped_countries.tsv' while being excluded from the calculations.
                          - Special/manual mappings are implemented for countries that cannot be detected via the script and is included/considered as high accuracy mappings (and included in the calculations).
                          - FluNet's and GISAID's countries are standardized by omitting any whitespace and are in lowercase formatting.
                          - Special inclusion of 'Unknown' country in the concatenated.tsv files but not plotted in the line graphs. 
                          - ERROR: Countries such as: Hong Kong (SAR) failed to be read properly in the FluNet's file and mapped with GISAID's 'Hong Kong (SAR)', whereas further data verifications for 'Papua New Guinea', 'Bolivia', and 'Lao' is needed.
                          - LIMITATIONS: FluNet's resolution is primarily limited to countries, making it incompatible to achieve higher-resolution mappings with GISAID's 'currLocation'.

    - DATA RANGE:
        - Selects the data range from datetime.now (as of 2025-07, end of July) up until 10 years back. 
        - Standardized the snapshot months/frequency by 1 month (going from 2025-07, 2025-06, and so on as the month_ends).
        - The time-window used for the sliding/moving window technique is 200 days, standardizing the window_end to the LAST DATE of EACH month (frequency = 1 month) and going -200 days as the window_start.
        - ISO_WEEKSTARTDATE:
            - Standardized gisaid_df['currCollectiondate'] to flunet_df['ISO_WEEKSTARTDATE'].
            - In instance for weeks (belonging to 2 overlapping months i.e. end-June and early-July), the data is always converted to the Monday of that particular week that belongs to WHICH month.
                -> The date format of ISO_WEEKSTATRTDATE always starts on Monday. This logic is applied for 'Collection_Year_Week' in the output file and how the dataset is categorized (in a weekly manner).
                - OUTPUT FILE:
                    ['GISAID_Country', 'FluNet_Country', 'Collection_Year_Month', 'Flunet_ISO_WEEK_STARTDATE', '#GISAID_Samples', '#FluNet_Cases', '#FluNet_Cases/#GISAID_Samples', 'Smoothed'200_days_#FluNet_Cases/#GISAID_Samples' 'Smoothed_#FluNet_Cases/#GISAID_Samples', '200_days_smoothed_#FluNet_Cases/#GISAID_Samples', 'Yearly_Median_#Cases/#Samples', 'Accession_IDs'] 
                    - '#GISAID_Samples' and '#FluNet_Cases' are the monthly sum/aggregation of samples and cases within each month
                    - '200_days_#FluNet_Cases/#GISAID_Samples' is the results generated when 200-days sliding windows is applied (generating 30+ values throughout the 30+ weeks, from present date until -200 days)
                    - The '200_days_#FluNet_Cases/#GISAID_Samples' contain the raw values of the average whereas the '200_days_smoothed_#FluNet_Cases/#GISAID_Samples' contain the smoothed values (with alpha = 0.3). 
                    - 'Accession_IDs' follows the number/count of #GISAID_Samples detected on a monthly snapshot (for every month, not when 200-days sliding window is applied). Any accession IDs are not repeated in different rows or 'Collection_Year_Month' (should be unique - to verify the accuracy).

    - SUBTYPE MAPPING:
        - GISAID ('A/H1') and 'currHost' is Human == FluNet ('AH1N12009') - as the data collected spans from 2025 to 2015, so it is generalized that all FluNet's data is attributed to humans as the primary infection hosts and all GISAID A/H1 that is mapped to AH1N1pdm200 (caused by the influenza outbreak/pandemic from 2009 onwards).
        - GISAID ('A/H3N2') and 'currHost' is Human == FluNet ('AH3') - as almost 95% of AH3 belongs to the 'A/H3N2' subtype
        - GISAID ('B/Victoria') 'currHost' is Human == FluNet ('BVIC') - with additional 'BVIC' sampling and cases counts are taken from:
            -> flunet_df ['BNOTDETERMINED'] and gisaid_df['currLineage':'B'] that are added into the '#FluNet_Cases' and '#GISAID_Samples' for B/Victoria and B/Yamagata (or BVIC and BYAM). 
            -> The additional '#FluNet_Cases' from 'BNOTDETERMINED' is classified/added into both 'BVIC' and 'BYAM' depending on the proportion/ratio of gisaid_df['B/Victoria'] and gisaid_df['B/Yamagata'] over 60-days window snapshot.
            -> The 60-days monthly snapshot is only used to acquire the ratio/proportion of 'BVIC' and 'BYAM' subtypes within the 'BNOTDETERMINED' subtype.

    - ALPHA SMOOTHING:
	- Coefficient is generalized with a value of 0.3 -> ".ewm(alpha=0.3, adjust=False)".
	- Dynamic alpha smoothing (0.3 and 0.5) is omitted from the script (used as a special 		treatment after the outliers are filtered out. 

'weighted_location_plotter_smoothed.py' 
    - Read the output files (in '.tsv' based on the different subtypes) - load and validate the data
    - x-axis contains the time window period from 2015-07 up until 2025-07 (present day, with freq = 'M')
    - The outputs for EACH subtype (with .tsv files imported from 'all_countries_*.tsv') are separated into: (1) '*_200_days_smoothed_weighted_location.png' and (2) '*_smoothed_monthly_weighted_location.png'
    - 'trends_analysis_report_smoothed.txt' is extracted by the function, 'def generate_accuracy_report(self, all_metrics):', to draw the comparison of the original (raw data) and the smoothed values (based on the means, std, and printing out the most smoothed countries - for each subtypes)
    - To identify the most significant peaks (in countries - for every subtypes - using 'scipy.signal.find_peaks()'). Only label the 'significant_countries_*' with the highest peaks in y-axis values - but some peaks are not identified the countries' name (requires further optimization).
    - Omit the country labelled as 'Unknown' from the plotted line-graph and used different HEX (colours) to represent each country, with legends attempting to show the peak/maximum y-value with its corresponding x-value. 

*** NOTE: Omitted sklearn.metrics (and the accuracy metrics) from the script
    



        


