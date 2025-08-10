import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import re
from typing import List, Tuple, Dict, Optional
from pandas.api.extensions import ExtensionArray

import warnings
warnings.filterwarnings('ignore')

import requests
# from requests.auth import HTTPBasicAuth
# from bs4 import BeautifulSoup

# Automate the download of FluNet's data - originally labelled as 'VIW_FNT.csv' 
def download_flunet_data(output_dir):
    flunet_file_url = "https://xmart-api-public.who.int/FLUMART/VIW_FNT?$format=csv"
    output_file = os.path.join(output_dir, "flunet_metadata.tsv") # Create the downloaded file (save it in the output directory for file reading)
    
    try:
        print("Downloading FluNet data...")
        response = requests.get(flunet_file_url, stream=True)
        response.raise_for_status()
        
        # Save the downloaded file as CSV 
        with open(output_file, "wb") as f:
            f.write(response.content)
            
        print(f"FluNet data downloaded successfully to: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error downloading FluNet data: {e}")
        return None

# Download FluNet data to the output directory
flunet_file = download_flunet_data(r"/Users/willnath/Desktop/Analysis #2")
if not flunet_file:
    flunet_file = r"/Users/willnath/Desktop/Analysis #2/flunet_metadata.tsv"

# Defining and reading the files
gisaid_file = r"/Users/willnath/Desktop/Analysis #2/accession2metadata.tsv"
flunet_file = r"/Users/willnath/Desktop/Analysis #2/flunet_metadata.tsv"
output_dir = r"/Users/willnath/Desktop/Analysis #2/all_countries_analysis_trial_crontab"

class EnhancedWeightedLocationAnalyzer:
    def __init__(self, window_size_days: int = 200, years_back: int = 10):
        self.gisaid_file = gisaid_file
        self.flunet_file = flunet_file
        self.output_dir = output_dir
        self.window_size_days = window_size_days
        self.years_back = years_back
        self.today = datetime.now()
        self.epsilon = 1e-9

        # Create the output directory if it does not exist
        os.makedirs(self.output_dir, exist_ok = True)

        print(f"EnhancedWeightedLocationAnalyzer initialized")
        print(f"Output directory: {self.output_dir}")
        print(f"Window size: {self.window_size_days} days")
        print(f"Years back: {self.years_back}")

        # Create empty list/dictionary for mapping (initialized)
        self.window_dates = []
        self.country_mappings = {} # Country mappings between GISAID and FluNet
        self.unmapped_countries = []

        # To define the FluNet sub-type column mapping (all columns are sourced from VIW_FNT.csv)
        self.flunet_subtype_columns = {
            'AH1': ['AH1', 'A_H1'],
            'AH1N12009': ['AH1N12009', 'AH1N1PDM09', 'A_H1N1PDM09'],
            'AH3': ['AH3', 'AH3N2', 'A_H3N2', 'A_H3'],
            'AH5': ['AH5', 'AH5N1', 'A_H5N1', 'A_H5'],
            'AH7N9': ['AH7N9', 'A_H7N9'],
            'AOTHER_SUBTYPE': ['AOTHER_SUBTYPE', 'A_OTHER', 'AOTHER'],
            'ANOTSUBTYPED': ['ANOTSUBTYPED', 'A_NOTSUBTYPED', 'ANOTSUBTYPABLE'],
            'BVIC': ['BVIC', 'B_VIC', 'BVIC_2DEL', 'BVIC_3DEL', 'BVIC_DELUNK', 'BVIC_NODEL'],
            'BYAM': ['BYAM', 'B_YAM', 'B_YAMAGATA'],
            'BNOTDETERMINED': ['BNOTDETERMINED', 'B_NOTDETERMINED', 'B_OTHER']
        }
    
    # Extract only the string of country values
    def extract_string_countries(self, df, column_name): # 'column_name' will be inputted with the column labels later on
        countries = []
        for value in df[column_name].dropna().unique(): 
            if isinstance(value, str):
                # Remove all numerical values from the string 
                cleaned_value = re.sub(r'\d+', ' ', value).strip() 
                if cleaned_value and len(cleaned_value) > 0:
                    countries.append(cleaned_value)
        return list(set(countries))
    
    # Standardize the country name (remove all whitespaces & all in lowercase)
    def standardize_country_name(self, country_name):
        if pd.isna(country_name) or country_name == ' ':
            return None
        return str(country_name).strip().lower()
    
    # To create mapping between countries in FluNet and GISAID
    def create_country_mappings(self, gisaid_df, flunet_df):
        print("\n === CREATING COUNTRY MAPPINGS ===")

        #Extract the string-only countries from both datasets
        print("\n(1) Extracting unique countries from GISAID's currCountry (string values only):")
        gisaid_countries = self.extract_string_countries(gisaid_df, 'currCountry') # Format: extract_string_countries(df, column_name)
        print(f"Found {len(gisaid_countries)} unique string countries from GISAID's currCountry:")
        for country in sorted(gisaid_countries):
            print(f" - {country}")

        print("\n(2) Extracting unique countries from FluNet 'COUNTRY_AREA_TERRITORY' (string values only):")
        flunet_countries = self.extract_string_countries(flunet_df, 'COUNTRY_AREA_TERRITORY')
        print(f"Found {len(flunet_countries)} unique string countries in FluNet:")
        for country in sorted(flunet_countries):
            print(f" - {country}")

        # Standardize GISAID and FluNet countries to all lowercase
        gisaid_countries_lower = [self.standardize_country_name(country) for country in gisaid_countries]
        flunet_countries_lower = [self.standardize_country_name(country) for country in flunet_countries]

        # Handle the special cases (or manual country mappings) that cannot be automatically detected via script
        # Format: 'FluNet Country Names':'Standardized Country Name'
        special_mappings = {
            'united kingdom': 'united kingdom',
            #'hong kong': 'hong kong (sar)',     
            'kosovo (in accordance with': 'kosovo', 
            'brunei darussalam': 'brunei',
            'venezuela (bolivarian republic of': 'venezuela',
            'viet nam': 'vietnam',
            'czechia': 'czech republic',
            #'china hong kong sar': 'hong kong (sar)',
            'China	 Hong Kong SAR': 'hong kong (sar)',
            'türkiye': 'turkey',
            'occupied palestinian territory': 'palestinian territory',
            'united republic of tanzania': 'tanzania',
            'united states of america': 'united states',
            'iran (islamic republic of': 'iran',
            'republic of korea': 'korea',
            'democratic people\'s republic of korea': 'korea',
            'papua new guinea': 'papua new guinea',
            'north macedonia': 'macedonia',
            'occupied palestinian territory including east jerusalem': 'palestinian territory',
            'bolivia (plurinational state of': 'bolivia',
            'lao people\'s democratic republic': 'lao',
            'netherlands': 'netherlands (kingdom of the)',
            'congo': 'democratic republic of congo',
            'republic of congo': 'republic of congo',
            'hong kong (sar)': 'china, hong kong sar',
            'moldova': 'republic of moldova',
            'libyan arab jamahiriya': 'libya',
            'bolivia': 'bolivia (plurinational state of)',
            'lao': "lao people's democratic republic"
        }
        

        # Apply special mappings to FluNet countries
        flunet_countries_processed = []
        for country in flunet_countries_lower:
            if country is None:
                continue
            processed_country = country

            # Implement the special mappings in FluNet (as 'replacement')
            for pattern, replacement in special_mappings.items():
                if pattern in country:
                    processed_country = replacement
                    break
            flunet_countries_processed.append(processed_country)
        
        # Add specific GISAID countries that might not be in FluNet but need mapping
        # These map GISAID country names to the standardized versions used by FluNet
        # Format: 'GISAID Country Names': 'Standardized Country Name'
        specific_gisaid_mappings = {
            'brunei': 'brunei',
            'venezuela': 'venezuela', 
            'vietnam': 'vietnam',
            'czech republic': 'czech republic',
            'hong kong (sar)': 'hong kong (sar)',
            'turkey': 'turkey',
            'palestinian territory': 'palestinian territory',
            'tanzania': 'tanzania',
            'united states': 'united states',
            'iran': 'iran',
            'korea': 'korea',
            'papua new guinea': 'papua new guinea',
            'macedonia': 'macedonia',
            'lao': 'lao',
            'netherlands': 'netherlands (kingdom of the)',
            'congo': 'democratic republic of congo',
            'republic of congo': 'republic of congo',
            'moldova': 'republic of moldova',
            'libya': 'libya', 
            'bolivia': 'bolivia' 
        }

        # Adding the specific GISAID countries into the processing list
        for gisaid_country, flunet_equivalent in specific_gisaid_mappings.items():
            if gisaid_country not in gisaid_countries_lower:
                gisaid_countries_lower.append(gisaid_country)

        # Mapping accuracy scores (only include >= 70% accuracy, accuracy_score = matched_words/total_words
        # Initialize the dictionary/list
        mapping_results = []
        high_accuracy_mappings = {}
        low_accuracy_mappings = {}

        for gisaid_country in gisaid_countries_lower:
            # Initialized all components
            best_match = None 
            best_score = 0
            matched_words = 0
            total_words = 0

             # First check if there's an exact special mapping
            if gisaid_country in specific_gisaid_mappings:
                exact_match = specific_gisaid_mappings[gisaid_country]
                if exact_match in flunet_countries_processed:
                    high_accuracy_mappings[gisaid_country] = {
                        'flunet_match': exact_match,
                        'accuracy_score': 100.0,
                        'match_type': 'high_accuracy'
                    }
                    continue
            

            for flunet_country in flunet_countries_processed:
                if gisaid_country is None or flunet_country is None:
                    continue

                # Split into words to assess accuracy scores (and comparison)
                gisaid_words = set(gisaid_country.split())
                flunet_words  = set(flunet_country.split())

                # To count the matching words (ratio of identical words from GISAID to FluNet's divided by FluNet's total words - for country names)
                common_words = gisaid_words.intersection(flunet_words)
                matched_words = len(common_words)
                total_words = len(flunet_words)

                if total_words > 0:
                    accuracy_score = (matched_words / total_words) * 100

                    if accuracy_score > best_score:
                        best_score = accuracy_score
                        best_match = flunet_country

            if best_match and best_score > 0:
                mapping_info = {
                    'gisaid_country': gisaid_country,
                    'flunet_country':  best_match,
                    'accuracy_score': best_score,
                    'matched_words': matched_words,
                    'total_words': total_words
                }

                mapping_results.append(mapping_info)

                # To filter out/separate the high and low accuracy mappings (threshold = 70%)
                if best_score >= 70:
                    high_accuracy_mappings[gisaid_country] = {
                        'flunet_match': best_match,
                        'flunet_country': best_match,
                        'accuracy_score': best_score,
                        'match_type': 'high_accuracy'
                    }
                else:
                    low_accuracy_mappings[gisaid_country] = {
                        'flunet_match': best_match,
                        'flunet_country': best_match,
                        'accuracy_score': best_score,
                        'match_type': 'low_accuracy'
                    }

        # Print out the mapping results
        print("\n(5) MAPPING RESULTS:")
        print(f"Successfully mapped countries (with more than 70% accuracy): {len(high_accuracy_mappings)}")
            
        for gisaid_country, mapping in high_accuracy_mappings.items():
            escaped_match = re.escape(mapping['flunet_match'])
            flunet_count = len(flunet_df[flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains(escaped_match, na=False, regex=True)])
            print(f" - {gisaid_country} == {mapping['flunet_match']} (accuracy: {mapping['accuracy_score']:.1f}%, FluNet rows = {flunet_count})")

        print(f"\nLow accuracy mappings (< 70% mapping): {len(low_accuracy_mappings)}")
        for gisaid_country, mapping in low_accuracy_mappings.items():
            escaped_match = re.escape(mapping['flunet_match'])
            flunet_count = len(flunet_df[flunet_df['COUNTRY_AREA_TERRITORY'].str.strip().str.lower().str.contains(escaped_match, na=False, regex=True)])
            print(f" - {gisaid_country} == {mapping['flunet_match']} (accuracy: {mapping['accuracy_score']:.1f}%, FluNet rows = {flunet_count})")

        # Identify the unmapped countries
        mapped_gisaid = set(high_accuracy_mappings.keys()) | set(low_accuracy_mappings.keys())
        unmapped_gisaid = set(gisaid_countries_lower) - mapped_gisaid

        mapped_flunet = set()
        for mapping in mapping_results:
            if mapping['accuracy_score'] >= 70:
                mapped_flunet.add(mapping['flunet_country'])

        unmapped_flunet = set(flunet_countries_processed) - mapped_flunet

        print(f"\nUnmapped GISAID countries: {len(unmapped_gisaid)}")
        for country in sorted([country for country in unmapped_gisaid if country is not None]):
            print(f" - {country}")

        print(f"\nUnmapped FluNet countries: {len(unmapped_flunet)}")
        for country in sorted(unmapped_flunet):
            if country is not None:
                escaped_country = re.escape(country)
                flunet_count = len(flunet_df[flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains(escaped_country, na=False, regex=True)])
                print(f" - {country} (FluNet rows: {flunet_count})")

        # Save the high-accuracy mappings (main output) into separate files
        if high_accuracy_mappings:
            high_accuracy_list = []
            for gisaid_country, mapping in high_accuracy_mappings.items(): # To define all the columns (and values) in the output file
                high_accuracy_list.append({
                    'gisaid_country': gisaid_country,
                    'flunet_country': mapping['flunet_match'],
                    'accuracy_score': mapping['accuracy_score'],
                    'match_type': mapping['match_type']
                    })

            # Defining the output file, saving the high-accuracy mappings into the output directory
            high_accuracy_df = pd.DataFrame(high_accuracy_list)
            high_accuracy_file = os.path.join(self.output_dir, "high_accuracy_country_mappings.tsv")
            os.makedirs(os.path.dirname(high_accuracy_file), exist_ok = True)
            high_accuracy_df.to_csv(high_accuracy_file, sep='\t', index=False)
            print(f"High accuracy mappings saved to: {high_accuracy_file}")

            # Low accuracy mappings (save to separate output)
            if low_accuracy_mappings:
                low_accuracy_list = []
                for gisaid_country, mapping in low_accuracy_mappings.items():
                    low_accuracy_list.append({
                        'gisaid_country': gisaid_country,
                        'flunet_country': mapping['flunet_match'],
                        'accuracy_score': mapping['accuracy_score'],
                        'match_type': mapping['match_type']
                        })

                low_accuracy_df = pd.DataFrame(low_accuracy_list)
                low_accuracy_file = os.path.join(self.output_dir, "low_accuracy_country_mappings.tsv")
                os.makedirs(os.path.dirname(low_accuracy_file), exist_ok = True)
                low_accuracy_df.to_csv(low_accuracy_file, sep='\t', index=False)
                print(f"Low accuracy mappings saved to: {low_accuracy_file}")

        # Saving the file for unmapped countries
        unmapped_list = []
                
        # Unmapped countries from GISAID
        for country in unmapped_gisaid:
            unmapped_list.append({
                'country': country,
                'source': 'GISAID',
                'reason': 'no_mapping_found'
                })

        # Unmapped countries from FluNet
        for country in unmapped_flunet:
            unmapped_list.append({
                'country': country,
                'source': 'FluNet',
                'reason': 'no_mapping_found'
                })

        if unmapped_list:
            unmapped_df = pd.DataFrame(unmapped_list)
            unmapped_file = os.path.join(self.output_dir, "unmapped_countries.tsv")
            os.makedirs(os.path.dirname(unmapped_file), exist_ok=True)
            unmapped_df.to_csv(unmapped_file, sep='\t', index=False)
            print(f"Unmapped countries saved to: {unmapped_file}")

        # Store the mapping to use for analysis part
        self.country_mappings = high_accuracy_mappings # Only include/use the high accuracy mappings
        self.unmapped_countries = unmapped_list

        return self.country_mappings
    
    def generate_moving_windows(self):
        # Generate the moving window with monthly snapshot dates (from last date of each month), for the last N years, with 200-day windows
        self.window_dates = []
        end_date = self.today.replace(day = 1) # Replace the end_date with the last day of each month
        start_date = end_date - timedelta(days = 365 * self.years_back) # years_back: int = 10
        months_ends = pd.date_range(start = start_date, end = end_date, freq = 'M') # The monthly snapshot

        for i, snapshot_date in enumerate(reversed(months_ends)):
            # To create the 200-days window ending at the last day of each month (from the most recent month, reversed)
            window_start = snapshot_date - timedelta(days = self.window_size_days) # snapshot_date is the end_date with an interval of 1 month (reversed from 2025-07 to 2015-07) 
            # window_start is the start_date (with window_size_days: int = 200 -> 200 days window size)

            # 'Collection-Year-Month' is based on the month of the snapshot (or window_end)
            collection_year_month = snapshot_date.strftime('%Y-%m')

            # Mapping the columns with values (printed out in terminals, not in output_file)
            self.window_dates.append({
                'window_id': i,
                'start_date': window_start,
                'end_date': snapshot_date,
                'snapshot_date': snapshot_date,
                'collection_year_month': collection_year_month
            })

            print(f"Generated {len(self.window_dates)} moving windows (monthly, {self.window_size_days} days each)")

    # Mapping subtypes of GISAID and FluNet (covers all subtypes)
    def map_gisaid_to_flunet_subtype(self, row):
        # Defining from GISAID's dataset
        lineage = str(row.get('currLineage', '')).strip()
        host = str(row.get('currHost', '')).strip().lower() # currHost needs to be 'human'

        # All A/H1N1 in GISAID's data (from 2009 onwards) is mapped into FluNet's AH1N12009
        if lineage == 'A/H1N1' and host == 'human':
            return 'AH1N12009' 
        if lineage == 'A/H3N2' and host == 'human':
            return 'AH3'
        if lineage == 'A/H7N9' and host == 'human':
            return 'AH7N9'
        if lineage.startswith('A/') and host == 'human' and not any(lineage.startswith(prefix) for prefix in ['A/H1', 'A/H3', 'A/H5', 'A/H7N9']):
            return 'AOTHER_SUBTYPE'
        if 'victoria' in lineage.lower() and host == 'human':
            return 'BVIC'
        if 'yamagata' in lineage.lower() and host == 'human':
            return 'BYAM'
        if lineage.startswith('B') and host == 'human' and not any(prefix in lineage.lower() for prefix in ['victoria', 'yamagata']):
            return 'BNOTDETERMINED' # More defined with the proportion of BVIC and BYAM is defined in later data's structure
        return 'UNKNOWN'
    
    # Loading GISAID data (accession2metadata.tsv)
    def load_gisaid_data(self) -> pd.DataFrame:
        print("Loading GISAID's database")
        if not os.path.exists(self.gisaid_file):
            print(f"Error: GISAID file cannot be found: {self.gisaid_file}")
            return pd.DataFrame()
        
        gisaid_columns = ['thisAccession', 'currIsolatename', 'currCollectiondate', 'currLocation',
            'currCountry', 'currCountryCode', 'currContinent', 'currLatitude',
            'currLongitude', 'combineMuts', 'currSubmissiondate', 'currCovClade',
            'currLineage', 'currOrigLab', 'currSubmLab', 'currSubmLabAddr',
            'currSeqTech', 'currAssemblyMethod', 'currHost', 'currGender',
            'currPatientAge', 'currPatientStatus', 'currLastVaccinated',
            'currSamplingStrategy', 'currAddLocationInfo', 'currAddHostInfo',
            'currLocationOri', 'currIsComplete']
        
        try:
            gisaid_df = pd.read_csv(
                self.gisaid_file,
                sep='\t',
                names=gisaid_columns,
                header=None,
                low_memory=False,
                encoding='utf-8',
                na_values=['NA', 'N/A', 'None', ' ', 'null', 'NULL'],
                keep_default_na=True,
                dtype=str)
        
            # Convert and standardize all the date columns in GISAID
            gisaid_df['currCollectiondate'] = pd.to_datetime(gisaid_df['currCollectiondate'], errors = 'coerce')
            gisaid_df['currSubmissiondate'] = pd.to_datetime(gisaid_df['currSubmissiondate'], errors='coerce')
            
            # Clean all the string columns in GISAID
            gisaid_df['currCountry'] = gisaid_df['currCountry'].fillna("Unknown").str.strip()
            gisaid_df['currLocation'] = gisaid_df['currLocation'].fillna("Unknown").str.strip()
            gisaid_df['currLineage'] = gisaid_df['currLineage'].fillna("Unknown").str.strip()
            gisaid_df['currHost'] = gisaid_df['currHost'].fillna("Unknown").str.strip()

            # Add ISO_WEEKSTARTDATE (counted from the Monday of weeks) and 'Collection_Year_Month' columns
            gisaid_df['ISO_WEEKSTARTDATE'] = gisaid_df['currCollectiondate'].apply(
                lambda d: d - pd.Timedelta(days = d.weekday()) if pd.notnull(d) else pd.NaT
            )

            gisaid_df['Collection_Year_Month'] = gisaid_df['ISO_WEEKSTARTDATE'].dt.strftime('%Y-%m')

            print(f"Loaded {len(gisaid_df)} GISAID records")
            return gisaid_df
        
        except Exception as e:
            print(f"Error loading GISAID data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
        
    def load_flunet_data(self) -> pd.DataFrame:
        # Load FluNet database
        print("Loading FluNet case count data")

        try:
            flunet_df = pd.read_csv(
                self.flunet_file,
                sep=',', # FluNet columns are separated by ',' for .csv file
                engine='python',
                na_values=[' ', 'NA', 'N/A', 'NULL', 'null', 'None', 'none'],
                quoting = 3,
                on_bad_lines = 'skip',
                dtype = str
            )
        
            # Strip whitespace from column names
            flunet_df.columns = flunet_df.columns.str.strip()

            # Debug print to show columns
            print("FluNet columns after loading:", flunet_df.columns.tolist())

            # Convert date column
            flunet_df['ISO_WEEKSTARTDATE'] = pd.to_datetime(flunet_df['ISO_WEEKSTARTDATE'], errors='coerce')
            
            # Clean country column
            flunet_df['COUNTRY_AREA_TERRITORY'] = flunet_df['COUNTRY_AREA_TERRITORY'].fillna("Unknown").str.strip()
            
            print(f"Loaded {len(flunet_df)} FluNet records")
            print(f"FluNet columns: {list(flunet_df.columns)}")
            return flunet_df
            
        except Exception as e:
            print(f"Error loading FluNet data: {e}")
            return pd.DataFrame()
        
    # To properly locate the FluNet's subtype columns (according to the mapping) for the calculation
    def find_flunet_subtype_column(self, flunet_df, subtype):
        possible_columns = self.flunet_subtype_columns.get(subtype, [subtype])

        for col_name in possible_columns:
            if col_name in flunet_df.columns:
                return col_name
            
        # Partial matching is used (if there is no exact match)
        for col in flunet_df.columns:
            if any(possible_col.lower() in col.lower() for possible_col in possible_columns):
                return col
            
        return None
    
    # Filter the GISAID and FluNet data based on the country-mapping (and accuracy score)
    def filter_data_by_country(self, gisaid_df, flunet_df, country_mapping):
        flunet_match = country_mapping['flunet_match']
        gisaid_country_match = country_mapping.get('gisaid_country_match', None)

        # Filter the GISAID data using the mapped country names
        gisaid_filter = pd.Series(False, index=gisaid_df.index)

        # Filter by currCountry (standardized to lowercase)
        if gisaid_country_match:
            gisaid_filter |= (gisaid_df['currCountry'].str.lower() == gisaid_country_match)
        else:
            # Use the mapping key as the country name
            gisaid_filter |= (gisaid_df['currCountry'].str.lower() == flunet_match)

        gisaid_country_df = gisaid_df[gisaid_filter].copy()

        # Filter the FluNet data to handle the special cases
        flunet_filter = pd.Series(False, index=flunet_df.index)

        # Handle the special cases for FluNet data
        if flunet_match == 'united kingdom':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.startswith('united kingdom')
        elif flunet_match == 'hong kong (sar)': # Failed to map
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('hong kong')
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('china')
        elif flunet_match == 'kosovo':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('kosovo')
        elif flunet_match == 'korea':
            # Sum both Republic of Korea and Democratic People's Republic of Korea (not specified in FluNet)
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('republic of korea')
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('democratic people\'s republic of korea')
        elif flunet_match == 'palestinian territory':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('palestinian territory')
        elif flunet_match == 'iran':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('iran')
        elif flunet_match == 'venezuela':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('venezuela')
        elif flunet_match == 'tanzania':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('tanzania')
        elif flunet_match == 'united states':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('united states')
        elif flunet_match == 'turkey':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('türkiye')
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('turkey')
        elif flunet_match == 'czech republic':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('czechia')
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('czech republic')
        elif flunet_match == 'macedonia':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('north macedonia')
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('macedonia')
        elif flunet_match == 'vietnam':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('viet nam')
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('vietnam')
        elif flunet_match == 'brunei':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('brunei')
        elif flunet_match == 'papua new guinea':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('papua new guinea')
        elif flunet_match == 'bolivia':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('bolivia')
        elif flunet_match == 'lao people\'s democratic republic':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('lao people\'s democratic republic')
        elif flunet_match == 'libya':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('libya')
        elif flunet_match == 'republic of congo':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('republic of congo')
        elif flunet_match == 'moldova':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('moldova')
        elif flunet_match == 'congo':
            flunet_filter |= flunet_df['COUNTRY_AREA_TERRITORY'].str.lower().str.contains('congo')
        else:
            flunet_filter |= (flunet_df['COUNTRY_AREA_TERRITORY'].str.lower() == flunet_match)

        flunet_country_df = flunet_df[flunet_filter].copy()

        return gisaid_country_df, flunet_country_df
    
    # Analyze the subtype for each country (raw and smoothed values)
    def analyze_subtype_for_country(self, gisaid_country_df, flunet_country_df, subtype, country_name, flunet_country_name=None):
    
        try:
            print(f"      Starting analysis for subtype: {subtype}")
            print(f"      GISAID country data shape: {gisaid_country_df.shape}")
            print(f"      FluNet country data shape: {flunet_country_df.shape}")
            
            # To map 'A/H1N1' to 'AH1N12009' as all GISAID's data as the time-window starts from 2015 onwards
            if subtype == 'AH1N12009':
                gisaid_subtype = gisaid_country_df[
                    (gisaid_country_df['currLineage'] == 'A/H1N1') & (gisaid_country_df['currHost'].str.lower() == 'human')].copy()
            else:
                gisaid_subtype = gisaid_country_df[
                    gisaid_country_df['flunet_subtype'] == subtype
                ].copy()
            
            print(f"      GISAID subtype data shape: {gisaid_subtype.shape}")
            print(f"      Unique lineages in subtype: {gisaid_subtype['currLineage'].unique()[:5]}")  # Show first 5 rows
            
            if len(gisaid_subtype) == 0:
                print(f"      No GISAID data found for subtype {subtype}")
                return pd.DataFrame()
            
            # Find FluNet column for this subtype (used the above function)
            flunet_column = self.find_flunet_subtype_column(flunet_country_df, subtype)
            print(f"      FluNet column found: {flunet_column}")

            if flunet_column and flunet_column not in flunet_country_df.columns:
                print(f"      Warning: FluNet column '{flunet_column}' not found in FluNet data")
                flunet_column = None
            
            # Only include months from NOW to -10 years ago
            end_month = self.today.replace(day=1)
            start_month = end_month - pd.DateOffset(years=self.years_back)
            valid_months = pd.date_range(start=start_month, end=end_month, freq='MS').strftime('%Y-%m')

            # Monthly aggregation (no overlaps of 'Accession_IDs, the number matches the '#GISAID_Samples')
            monthly_results = []
            for month in valid_months:
                month_end = pd.to_datetime(month) + pd.offsets.MonthEnd(0)

                # Generate the 60-days windows with ratio of BVIC and BYAM
                ratio_window_start = month_end - pd.Timedelta(days=59)
                ratio_window_end = month_end

                # GISAID: Count B/Victoria and B/Yamagata in 60-day window for the ratio
                gisaid_60d = gisaid_country_df[(gisaid_country_df['ISO_WEEKSTARTDATE'] >= ratio_window_start) & (gisaid_country_df['ISO_WEEKSTARTDATE'] <= ratio_window_end)]
                bvic_count = len(gisaid_60d[gisaid_60d['flunet_subtype'] == 'BVIC']['thisAccession'].dropna())
                byam_count = len(gisaid_60d[gisaid_60d['flunet_subtype'] == 'BYAM']['thisAccession'].dropna())
                total_bvic_byam = bvic_count + byam_count

                # Initialized the total cases of BVIC and BYAM (if the total cases are 0, the ratio is made 50% each)
                if total_bvic_byam == 0:
                    ratio_bvic = 0.5
                    ratio_byam = 0.5
                else:
                    ratio_bvic = bvic_count / total_bvic_byam
                    ratio_byam = byam_count / total_bvic_byam

                # GISAID - Sum all the counts ONLY for this month (for monthly snapshot, prior to applying the 200-days moving time-window)
                # Creates the mask (for each month in valid_months, the code sets month_end to the last day of the month)
                # Ensures that the weeks (within the overlapping months) are assigned to the month containing the Monday of that particular week
                gisaid_mask = gisaid_subtype['ISO_WEEKSTARTDATE'].dt.month == month_end.month
                gisaid_mask &= gisaid_subtype['ISO_WEEKSTARTDATE'].dt.year == month_end.year
                gisaid_month = gisaid_subtype[gisaid_mask]

                # Accessions are sourced from GISAID, starts with 'EPI_ISL'
                accessions = [acc for acc in gisaid_month['thisAccession'].dropna() if acc.startswith('EPI_ISL_')]
                num_gisaid_samples = len(accessions)
                accession_str = ','.join(accessions)

                # GISAID: Count 'B' (not Victoria/Yamagata) in this month
                gisaid_b_mask = (gisaid_country_df['flunet_subtype'] == 'BNOTDETERMINED') & (gisaid_country_df['ISO_WEEKSTARTDATE'].dt.month == month_end.month) & (gisaid_country_df['ISO_WEEKSTARTDATE'].dt.year == month_end.year)
                gisaid_b_samples = len(gisaid_country_df[gisaid_b_mask]['thisAccession'].dropna())

                # FluNet: All records for this month
                flunet_mask = flunet_country_df['ISO_WEEKSTARTDATE'].dt.month == month_end.month
                flunet_mask &= flunet_country_df['ISO_WEEKSTARTDATE'].dt.year == month_end.year

                flunet_cases = 0
                bnotdetermined_cases = 0

                if flunet_column:
                    flunet_cases_series = pd.to_numeric(
                        flunet_country_df.loc[flunet_mask, flunet_column],
                        errors='coerce'
                    )
                    if isinstance(flunet_cases_series, pd.Series):
                        flunet_cases = flunet_cases_series.fillna(0).sum()

                # Add BNOTDETERMINED cases proportionally for BVIC and BYAM
                if subtype in ['BVIC', 'BYAM']:
                    bnot_col = self.find_flunet_subtype_column(flunet_country_df, 'BNOTDETERMINED')
                    if bnot_col and bnot_col in flunet_country_df.columns:
                        bnot_series = pd.to_numeric(flunet_country_df.loc[flunet_mask, bnot_col], errors='coerce')
                        if isinstance(bnot_series, pd.Series):
                            bnotdetermined_cases = bnot_series.fillna(0).sum()
            
                    # Add GISAID 'B' samples for this month (based on the monthly snapshot)
                    gisaid_b_add = gisaid_b_samples * (ratio_bvic if subtype == 'BVIC' else ratio_byam)
                    num_gisaid_samples += gisaid_b_add

                    # Add BNOTDETERMINED FluNet cases proportionally
                    flunet_cases += bnotdetermined_cases * (ratio_bvic if subtype == 'BVIC' else ratio_byam)
                ratio = flunet_cases / num_gisaid_samples if num_gisaid_samples > 0 else 0
                monthly_results.append({
                    'GISAID_Country': country_name,
                    'FluNet_Country': flunet_country_name if flunet_country_name else country_name,
                    'Collection_Year_Month': month,
                    'Flunet_ISO_WEEK_STARTDATE': month_end - pd.Timedelta(days=month_end.weekday()),
                    '#GISAID_Samples': num_gisaid_samples,
                    '#FluNet_Cases': flunet_cases,
                    '#FluNet_Cases/#GISAID_Samples': ratio,
                    'Accession_IDs': accession_str
                })

            monthly_df = pd.DataFrame(monthly_results)
            # 200-day sliding window aggregation (overlap allowed) 
            window_results = []

            # Reiterating end_month to the last day of the current month
            end_month = self.today.replace(day=1) + pd.offsets.MonthEnd(0)
            start_month = (end_month - pd.DateOffset(years=self.years_back)).replace(day=1) + pd.offsets.MonthEnd(0)
            window_ends = pd.date_range(start=start_month, end=end_month, freq='M')
            for window_end in window_ends: 
                window_start = window_end - pd.Timedelta(days=199)
                collection_year_month = window_end.strftime('%Y-%m')
                print(f"200-day window for {country_name} ({subtype}) {collection_year_month}: "
                      f"start_date={window_start.date()}, end_date={window_end.date()}, window_size=200D")
                
                # 60-day window for ratio calculation
                ratio_window_start = window_end - pd.Timedelta(days=59)
                ratio_window_end = window_end
                gisaid_60d = gisaid_country_df[(gisaid_country_df['ISO_WEEKSTARTDATE'] >= ratio_window_start) & (gisaid_country_df['ISO_WEEKSTARTDATE'] <= ratio_window_end)]
                bvic_count = len(gisaid_60d[gisaid_60d['flunet_subtype'] == 'BVIC']['thisAccession'].dropna())
                byam_count = len(gisaid_60d[gisaid_60d['flunet_subtype'] == 'BYAM']['thisAccession'].dropna())
                total_bvic_byam = bvic_count + byam_count
                if total_bvic_byam == 0:
                    ratio_bvic = 0.5
                    ratio_byam = 0.5
                else:
                    ratio_bvic = bvic_count / total_bvic_byam
                    ratio_byam = byam_count / total_bvic_byam
                # Filter data for the window
                gisaid_window = gisaid_subtype[
                    (gisaid_subtype['ISO_WEEKSTARTDATE'] >= window_start) & 
                    (gisaid_subtype['ISO_WEEKSTARTDATE'] <= window_end)
                ]
                flunet_window = flunet_country_df[
                    (flunet_country_df['ISO_WEEKSTARTDATE'] >= window_start) & 
                    (flunet_country_df['ISO_WEEKSTARTDATE'] <= window_end)
                ]

                # GISAID: count 'B' (not Victoria/Yamagata) in window
                gisaid_b_mask = (gisaid_country_df['flunet_subtype'] == 'BNOTDETERMINED') & (gisaid_country_df['ISO_WEEKSTARTDATE'] >= window_start) & (gisaid_country_df['ISO_WEEKSTARTDATE'] <= window_end)
                gisaid_b_samples = len(gisaid_country_df[gisaid_b_mask]['thisAccession'].dropna())

                # Weekly aggregation within the 200-days window based on weekly summation & ratio (30+ weeks/data points being averaged)
                gisaid_window_epi = gisaid_window[gisaid_window['thisAccession'].str.startswith('EPI_ISL_', na=False)]
                gisaid_weekly = gisaid_window_epi.groupby('ISO_WEEKSTARTDATE')['thisAccession'].nunique().reset_index()
                gisaid_weekly.rename(columns={'thisAccession': 'gisaid_samples_weekly'}, inplace=True)

                # FluNet: weekly counts
                if flunet_column and flunet_column in flunet_country_df.columns:
                    flunet_window_copy = flunet_window.copy()
                    flunet_window_copy[flunet_column] = pd.to_numeric(flunet_window_copy[flunet_column], errors='coerce').fillna(0)
                    flunet_weekly = flunet_window_copy.groupby('ISO_WEEKSTARTDATE')[flunet_column].sum().reset_index()
                    flunet_weekly.rename(columns={flunet_column: 'flunet_cases_weekly'}, inplace=True)
                else:
                    unique_weeks_in_gisaid = pd.DataFrame({'ISO_WEEKSTARTDATE': gisaid_weekly['ISO_WEEKSTARTDATE'].unique()})
                    if not unique_weeks_in_gisaid.empty:
                        flunet_weekly = unique_weeks_in_gisaid
                        flunet_weekly['flunet_cases_weekly'] = 0
                    else:
                         flunet_weekly = pd.DataFrame(columns=['ISO_WEEKSTARTDATE', 'flunet_cases_weekly'])

                # Add BNOTDETERMINED cases proportionally for BVIC and BYAM
                bnotdetermined_cases = 0
                if subtype in ['BVIC', 'BYAM']:
                    bnot_col = self.find_flunet_subtype_column(flunet_country_df, 'BNOTDETERMINED')
                    if bnot_col and bnot_col in flunet_country_df.columns:
                        bnot_series = pd.to_numeric(flunet_window[bnot_col], errors='coerce')
                        if isinstance(bnot_series, pd.Series):
                            bnotdetermined_cases = bnot_series.fillna(0).sum()

                    # Add GISAID 'B' samples for this window
                    gisaid_b_add = gisaid_b_samples * (ratio_bvic if subtype == 'BVIC' else ratio_byam)

                    # Add to weekly GISAID samples
                    gisaid_weekly['gisaid_samples_weekly'] += gisaid_b_add / max(len(gisaid_weekly), 1)

                    # Add BNOTDETERMINED FluNet cases proportionally
                    flunet_weekly['flunet_cases_weekly'] += bnotdetermined_cases * ((ratio_bvic if subtype == 'BVIC' else ratio_byam) / max(len(flunet_weekly), 1))

                # Merge weekly data (with the condition that the weekly, parsed datasets are not empty)
                if not gisaid_weekly.empty or not flunet_weekly.empty:
                    # Mapped/combined based on the 'ISO_WEEKSTARTDATE' of both datasets
                    weekly_data = pd.merge(gisaid_weekly, flunet_weekly, on='ISO_WEEKSTARTDATE', how='outer').fillna(0)
                    weekly_data['weekly_ratio'] = weekly_data['flunet_cases_weekly'] / weekly_data['gisaid_samples_weekly']
                    weekly_data['weekly_ratio'].replace([np.inf, -np.inf], np.nan, inplace=True) # Replace all +/- infinity with NaN value
                    avg_ratio_200d = weekly_data['weekly_ratio'].mean()
                    if pd.isna(avg_ratio_200d):
                        avg_ratio_200d = 0.0
                else:
                    avg_ratio_200d = 0.0
                
                # Overall window counts
                gisaid_window_accessions = [acc for acc in gisaid_window['thisAccession'].dropna() if acc.startswith('EPI_ISL_')]
                num_gisaid_samples_200d = len(set(gisaid_window_accessions))
                if subtype in ['BVIC', 'BYAM']:
                    num_gisaid_samples_200d += gisaid_b_samples * (ratio_bvic if subtype == 'BVIC' else ratio_byam)
                flunet_cases_200d = 0
                if flunet_column and flunet_column in flunet_country_df.columns:
                    flunet_cases_series = pd.to_numeric(flunet_window[flunet_column], errors='coerce')
                    if isinstance(flunet_cases_series, pd.Series):
                        flunet_cases_200d = flunet_cases_series.fillna(0).sum()
                if subtype in ['BVIC', 'BYAM']:
                    flunet_cases_200d += bnotdetermined_cases * (ratio_bvic if subtype == 'BVIC' else ratio_byam)
                window_results.append({
                    'Collection_Year_Month': collection_year_month,
                    'Flunet_ISO_WEEK_STARTDATE': window_end - pd.Timedelta(days=window_end.weekday()),
                    '200_days_#FluNet_Cases/#GISAID_Samples': avg_ratio_200d,
                    '200_days_Accession_IDs': ','.join(sorted(set(gisaid_window_accessions))),
                    '200_days_window_start': window_start.date(),
                    '200_days_window_end': window_end.date()
                })
            window_df = pd.DataFrame(window_results)

            # Fill NaN in the 200_days_#FluNet_Cases/#GISAID_Samples column with 0.0
            if '200_days_#FluNet_Cases/#GISAID_Samples' in window_df.columns:
                window_df['200_days_#FluNet_Cases/#GISAID_Samples'] = window_df['200_days_#FluNet_Cases/#GISAID_Samples'].fillna(0.0)

            # Merge monthly and window results on Collection_Year_Month
            merged_df = pd.merge(monthly_df, window_df, on=['Collection_Year_Month', 'Flunet_ISO_WEEK_STARTDATE'], how='outer')

            # Ensure 200-day columns are always >= monthly columns
            for col_pair in [("#GISAID_Samples", "200_days_#GISAID_Samples"), ("#FluNet_Cases", "200_days_#FluNet_Cases")]:
                monthly_col, window_col = col_pair
                if monthly_col in merged_df.columns and window_col in merged_df.columns:
                    merged_df[window_col] = merged_df[[monthly_col, window_col]].max(axis=1)

            # Group by Collection_Year_Month to ensure only one row per month
            def unique_concat(series):
                try:
                    if series.isnull().all():
                        return ''
                    # Handle both, the string and non-string series
                    if series.dtype == 'object':
                        return ','.join(sorted(set(','.join(series.dropna()).split(','))))
                    else:
                        return ','.join(sorted(set(str(x) for x in series.dropna())))
                except Exception as e:
                    print(f"      Warning: Error in unique_concat: {e}") # To detect any errors in concatenating the Accession_IDs
                    return ''

            agg_dict = {} # Initialized the aggregation dictionary - to efficiently aggregate the data and handle different types of columns (i.e. string, numeric, boolean, etc)
            for col in merged_df.columns:
                if col in ['Accession_IDs', '200_days_Accession_IDs']:
                    agg_dict[col] = unique_concat
                elif merged_df[col].dtype.kind in 'biufc' and col not in ['Year']:
                    agg_dict[col] = 'sum'
                else:
                    agg_dict[col] = 'first'
            
            try: # Merging/grouping by Collection_Year_Month, and applying the aggregation dictionary
                merged_df = merged_df.groupby('Collection_Year_Month', as_index=False).agg(agg_dict)
            except Exception as e:
                print(f"      Warning: Error in aggregation for {country_name}: {e}")
                print(f"      Attempting to handle aggregation manually...")
                # Fallback: try to handle aggregation  manually
                try:
                    merged_df = merged_df.groupby('Collection_Year_Month', as_index=False).agg('first')
                except Exception as e2:
                    print(f"      Critical: Aggregation failed completely: {e2}")
                    return pd.DataFrame()

            # Ensuring that merged_df is a DataFrame after aggregation
            if not isinstance(merged_df, pd.DataFrame):
                merged_df = pd.DataFrame(merged_df)

            # Ensuring that the columns are series for EWMA and other operations (with the condition that smoothing with alpha = 0.3 is applied))
            if '#FluNet_Cases/#GISAID_Samples' in merged_df.columns:
                merged_df['Smoothed_#FluNet_Cases/#GISAID_Samples'] = pd.Series(merged_df['#FluNet_Cases/#GISAID_Samples']).ewm(alpha=0.3, adjust=False).mean()
            if '200_days_#FluNet_Cases/#GISAID_Samples' in merged_df.columns:
                merged_df['200_days_smoothed_#FluNet_Cases/#GISAID_Samples'] = pd.Series(merged_df['200_days_#FluNet_Cases/#GISAID_Samples']).ewm(alpha=0.3, adjust=False).mean()

            # Yearly median (raw values) if Flunet_ISO_WEEK_STARTDATE column exists (to generate more precise outcomes)
            if 'Flunet_ISO_WEEK_STARTDATE' in merged_df.columns:
                year_series = pd.to_datetime(pd.Series(merged_df['Flunet_ISO_WEEK_STARTDATE']), errors='coerce').dt.year
                merged_df['Year'] = year_series.values
                if '#FluNet_Cases/#GISAID_Samples' in merged_df.columns:
                    merged_df['Yearly_Median_#FluNet_Cases/#GISAID_Samples'] = merged_df.groupby('Year')['#FluNet_Cases/#GISAID_Samples'].transform('median')
                merged_df = merged_df.drop(columns=['Year'])

            # Add any static columns
            merged_df['GISAID_Country'] = country_name
            merged_df['FluNet_Country'] = flunet_country_name if flunet_country_name else country_name

            # Defining the header (or labels) for the output file's columns - reordering the columns for the output file
            output_columns = [
                'GISAID_Country', 'FluNet_Country', 'Collection_Year_Month', 'Flunet_ISO_WEEK_STARTDATE',
                '#GISAID_Samples', '#FluNet_Cases', '#FluNet_Cases/#GISAID_Samples',
                '200_days_#FluNet_Cases/#GISAID_Samples',
                'Smoothed_#FluNet_Cases/#GISAID_Samples', '200_days_smoothed_#FluNet_Cases/#GISAID_Samples',
                'Yearly_Median_#FluNet_Cases/#GISAID_Samples', 'Accession_IDs'
            ]
            for col in output_columns:
                if col not in merged_df.columns:
                    merged_df[col] = np.nan
            merged_df = merged_df[output_columns]
            merged_df = merged_df.fillna(0.0)
            print(f"      Final DataFrame shape: {merged_df.shape}")
            print(f"      DataFrame columns: {merged_df.columns.tolist()}")
            return merged_df
                
        except Exception as e:
            print(f"      Critical error in analyze_subtype_for_country: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    # Implement all-the-above script
    def analyze_all_countries(self):
        print("=== STARTING COMPREHENSIVE ANALYSIS FOR ALL COUNTRIES ===")

        gisaid_df = self.load_gisaid_data()
        flunet_df = self.load_flunet_data()

        if gisaid_df.empty or flunet_df.empty:
            print("One or both datasets are empty. Exiting analysis.")
            return
        
        country_mappings = self.create_country_mappings(gisaid_df, flunet_df)
        if not country_mappings:
            print("No country mappings found. Exiting analysis.")
            return
        
        print("\n=== MAPPING GISAID SUBLINEAGES TO FLUNET SUBTYPES ===")
        gisaid_df['flunet_subtype'] = gisaid_df.apply(self.map_gisaid_to_flunet_subtype, axis=1)
        print(f"Unique flunet_subtype values after mapping: {gisaid_df['flunet_subtype'].unique()}")

        self.generate_moving_windows()

        all_results_by_subtype = {}
        country_summary = []
        accuracy_results = []

        for mapped_country, mapping in country_mappings.items():
            print(f"\n=== PROCESSING COUNTRY: {mapped_country} ===")
            gisaid_country_df, flunet_country_df = self.filter_data_by_country(
                gisaid_df, flunet_df, mapping
            )

            print(f"  GISAID records after country filter: {len(gisaid_country_df)}")
            print(f"  FluNet records after country filter: {len(flunet_country_df)}")

            if gisaid_country_df.empty:
                print(f"No GISAID data for {mapped_country}")
                continue
            if flunet_country_df.empty:
                print(f"No FluNet data for {mapped_country}")
                continue

            print(f"GISAID records: {len(gisaid_country_df)}")
            print(f"FluNet records: {len(flunet_country_df)}")

            unique_subtypes = gisaid_country_df['flunet_subtype'].unique()
            print(f"  Unique subtypes in GISAID for this country: {unique_subtypes}")

            unique_subtypes = [s for s in unique_subtypes if s != 'UNKNOWN']
            if 'AH1N12009' in gisaid_country_df['flunet_subtype'].unique():
                if 'AH1N12009' not in unique_subtypes:
                    unique_subtypes.append('AH1N12009')
            print(f"Subtypes found: {unique_subtypes}")
            country_results = []
            for subtype in unique_subtypes:
                if subtype == 'BNOTDETERMINED':
                    # Do not analyze or output BNOTDETERMINED as a separate subtype
                    continue
                try:
                    print(f"    Analyzing subtype: {subtype}")
                    subtype_results = self.analyze_subtype_for_country(
                        gisaid_country_df, flunet_country_df, subtype, mapped_country, mapping['flunet_match']
                    )
                    print(f"    Subtype results DataFrame shape: {subtype_results.shape}")

                    if not subtype_results.empty:
                        output_subtype = subtype

                        # Keep the original subtype name for file naming
                        output_subtype = subtype
                        country_results.append(subtype_results)

                        if output_subtype not in all_results_by_subtype:
                            all_results_by_subtype[output_subtype] = []

                        all_results_by_subtype[output_subtype].append(subtype_results)
                        accuracy_metrics = self.calculate_accuracy_metrics(subtype_results)
                        accuracy_metrics.update({
                            'country': mapped_country,
                            'subtype': output_subtype,
                            'total_sequences': subtype_results['#GISAID_Samples'].sum(),
                            'total_cases': subtype_results['#FluNet_Cases'].sum()
                        })

                    else:
                        print(f"    No data for subtype {subtype} in {mapped_country}")
                except Exception as e:
                    print(f"Error analyzing {subtype} for {mapped_country}: {e}")
                    continue

            if country_results:
                combined_country_results = pd.concat(country_results, ignore_index=True)
                country_summary.append({
                    'country': mapped_country,
                    'total_sequences': combined_country_results['#GISAID_Samples'].sum(),
                    'total_cases': combined_country_results['#FluNet_Cases'].sum(),
                    'subtypes_count': len(unique_subtypes),
                    'subtypes': ', '.join(unique_subtypes),
                    'windows_with_data': (combined_country_results['#GISAID_Samples'] > 0).sum(),
                    'match_type': mapping['match_type']
                })
        print("\n=== SAVING SUBTYPE RESULTS ===")
        files_written = 0
        for subtype, results_list in all_results_by_subtype.items():
            if subtype == 'BNOTDETERMINED':
                # No need for a separate outfile file for BNOTDETERMINED
                continue
            if results_list:
                combined_subtype_results = pd.concat(results_list, ignore_index=True)
                combined_subtype_results = combined_subtype_results.sort_values(
                    ['GISAID_Country', 'Flunet_ISO_WEEK_STARTDATE']
                )
                output_file = os.path.join(self.output_dir, f"all_countries_{subtype}.tsv")
                if subtype == 'AH1N12009':
                    output_file = os.path.join(self.output_dir, f"all_countries_AH1N12009.tsv")
                # Reorder columns for output
                output_columns = [
                    'GISAID_Country', 'FluNet_Country', 'Collection_Year_Month', 'Flunet_ISO_WEEK_STARTDATE',
                    '#GISAID_Samples', '#FluNet_Cases', '#FluNet_Cases/#GISAID_Samples',
                    '200_days_#FluNet_Cases/#GISAID_Samples',
                    'Smoothed_#FluNet_Cases/#GISAID_Samples', '200_days_smoothed_#FluNet_Cases/#GISAID_Samples',
                    'Yearly_Median_#FluNet_Cases/#GISAID_Samples', 'Accession_IDs'
                ]
                
                try:
                    combined_subtype_results = combined_subtype_results[output_columns]
                except KeyError as e:
                    print(f"[ERROR] Could not write {output_file} due to missing columns: {e}")
                    print(f"  DataFrame columns: {combined_subtype_results.columns.tolist()}")
                    continue
                combined_subtype_results.to_csv(output_file, sep='\t', index=False)
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                print(f"Saved {subtype} results to {output_file}")
                files_written += 1

        if files_written == 0:
            print("[WARNING] No subtype files were generated. Check your data and mapping logic.")

        if country_summary:
            summary_df = pd.DataFrame(country_summary)
            summary_file = os.path.join(self.output_dir, "country_summary.tsv")
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
            summary_df.to_csv(summary_file, sep='\t', index=False)
            print(f"Country summary saved to {summary_file}")
                
        print(f"\n=== ANALYSIS COMPLETE ===")
        print(f"Processed {len(country_mappings)} countries")
        print(f"Generated {len(all_results_by_subtype)} subtype files")
        print(f"Results saved to: {self.output_dir}")


def main():
    """Main function to run the analysis"""
    print("Enhanced Weighted Location Analyzer - All Countries Analysis")
    print("This will analyze all mapped countries automatically.")
    
    # Initialize analyzer
    analyzer = EnhancedWeightedLocationAnalyzer(
        window_size_days=200,
        years_back=10
    )

    # Run comprehensive analysis
    analyzer.analyze_all_countries()
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report()

if __name__ == "__main__":
    main()