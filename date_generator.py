# Function to read lines from a text file and return them as a list of strings
def read_lines_to_list(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Remove any trailing newline characters from each line
            lines = [line.strip() for line in lines]
        return lines
    except FileNotFoundError:
        print(f"The file {file_path} was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

# Example usage
file_path = '/home/owner/Documents/copernicus/arquivos'  # Replace with your file path
lines_list = read_lines_to_list(file_path)

# Print the list of lines
print(lines_list)

arq_dates=[]
for k in lines_list:
    arq_dates.append(k[15:25])


import datetime

def generate_dates1():
    date_list = []
    start_year = 2012
    end_year = 2012
    hours = ['00', '06', '12', '18']
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Get the number of days in the month (handling February for leap years)
            if month == 2:
                if (year % 400 == 0) or (year % 100 != 0 and year % 4 == 0):
                    max_day = 29
                else:
                    max_day = 28
            elif month in [4, 6, 9, 11]:
                max_day = 30
            else:
                max_day = 31
            
            for day in range(1, max_day + 1):
                for hour in hours:
                    # Format the date string with leading zeros
                    #date_str = f"{year}-{month:02d}-{day:02d}-{hour}" #format with dashes in between
                    date_str = f"{year}{month:02d}{day:02d}{hour}"
                    date_list.append(date_str)
    
    return date_list

# # Generate the dates
# all_dates = generate_dates()

def generate_dates():
    date_list = []
    start_year = 2011
    end_year = 2023
    hours = ['00', '06', '12', '18']
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            # Get the number of days in the month (handling February for leap years)
            if month == 2:
                if (year % 400 == 0) or (year % 100 != 0 and year % 4 == 0):
                    max_day = 29
                else:
                    max_day = 28
            elif month in [4, 6, 9, 11]:
                max_day = 30
            else:
                max_day = 31
            
            for day in range(1, max_day + 1):
                for hour in hours:
                    # Format the date string with leading zeros
                    #date_str = f"{year}-{month:02d}-{day:02d}-{hour}" #format with dashes in between
                    date_str = f"{year}{month:02d}{day:02d}{hour}"
                    date_list.append(date_str)
                    
    
    return date_list

# Generate the dates
all_dates = generate_dates()

# Example: Print the first 10 and last 10 dates to verify
print("First 10 dates:")
for date in all_dates[:300]:
    print(date)

print("\nLast 10 dates:")
for date in all_dates[-10:]:
    print(date)

# Print total number of dates generated
print(f"\nTotal dates generated: {len(all_dates)}")


missing_dates=[]
counter=0
for k in all_dates:
    if k in arq_dates:
        counter=counter+1
    else:
        missing_dates.append(k)
        
        
def write_list_to_file(filename, data_list):
    """
    Writes each element of a list to a new line in a text file.
    
    Args:
        filename (str): Name of the output file
        data_list (list): List of items to write to the file
    """
    with open(filename, 'w') as file:
        for item in data_list:
            file.write(f"{item}\n")  # \n creates a new line
        

write_list_to_file("/home/owner/Documents/copernicus/dates_list.txt", missing_dates)


from pandas import date_range
dts = date_range(
    '1979-01-01 00:00:00',
    '2011-03-31 18:00:00',
    freq='6H'
)


from datetime import datetime
def convert_date_format(input_date):
    """Convert from yyyymmddhh to yyyy-mm-dd hh:00:00 format"""
    try:
        # Parse the input date string
        dt = datetime.strptime(input_date.strip(), '%Y%m%d%H')
        # Format it to the desired output
        return dt.strftime('%Y-%m-%d %H:00:00')
    except ValueError as e:
        print(f"Error parsing date '{input_date}': {e}")
        return None

def read_and_convert_dates(file_path):
    """Read dates from file and return converted dates in a list"""
    converted_dates = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    converted = convert_date_format(line)
                    if converted is not None:
                        converted_dates.append(converted)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return converted_dates

# Example usage:
file_path = '/home/owner/Documents/copernicus/dates_list.txt'  # Replace with your file path
converted_dates = read_and_convert_dates(file_path)

# Print the results
for date in converted_dates:
    print(date)



######################################################################################
from datetime import datetime

def parse_date(input_date):
    """Parse yyyymmddhh string into a datetime object"""
    try:
        # Parse the input date string and return datetime object
        return datetime.strptime(input_date.strip(), '%Y%m%d%H')
    except ValueError as e:
        print(f"Error parsing date '{input_date}': {e}")
        return None

def read_and_parse_dates(file_path):
    """Read dates from file and return datetime objects in a list"""
    parsed_dates = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    dt = parse_date(line)
                    if dt is not None:
                        parsed_dates.append(dt)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return parsed_dates

# Example usage:
file_path = '/home/owner/Documents/copernicus/dates_list.txt'  # Replace with your file path
datetime_objects = read_and_parse_dates(file_path)

for dt in datetime_objects:
    print(dt)




















