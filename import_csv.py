import pandas as pd
import re
from sqlalchemy import create_engine, text
import os
from datetime import datetime
import numpy as np

# Database connection settings - update with your password
db_user = "postgres"
db_password = "your_password_here"  # Replace with your actual password
db_host = "localhost"
db_port = "5432"
db_name = "aircraft_maintenance"

# Create connection string
db_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_string)

def extract_tail_number(filename):
    """Extract SMK tail number from filename"""
    match = re.search(r'SMK(\d+)', filename)
    if match:
        return f"SMK{match.group(1)}"
    else:
        return "UNKNOWN"  # Fallback value

def process_csv(filepath, skip_rows=3):
    """Process a single CSV file"""
    try:
        # Get filename and extract tail number
        filename = os.path.basename(filepath)
        tail_number = extract_tail_number(filename)
        print(f"Processing file: {filename}")
        print(f"Extracted tail number: {tail_number}")
        
        # Read CSV file
        print(f"Reading CSV data (skipping {skip_rows} header rows)...")
        
        # Read the file with pandas - using low_memory=False to avoid mixed types warning
        df = pd.read_csv(filepath, skiprows=skip_rows, header=None, low_memory=False)
        print(f"Read {len(df)} rows from CSV")
        
        # Determine the actual number of columns
        num_columns = len(df.columns)
        print(f"CSV has {num_columns} columns")
        
        # Create column names based on the actual column count
        # Base column names (these are the ones we know)
        base_columns = [
            'Session Time', 'GPS Fix Quality', 'Number of Satellites', 'GPS Date & Time',
            'Latitude (deg)', 'Longitude (deg)', 'GPS Altitude (feet)', 'Ground Speed (knots)',
            'Ground Track (deg)', 'Mag Var (deg)', 'Cross Track Error (NM)', 
            'Destination Waypoint ID', 'Range to Destination (NM)', 'Bearing to Destination (deg)',
            'System Time', 'Pitch (deg)', 'Roll (deg)', 'Magnetic Heading (deg)',
            'Indicated Airspeed (knots)', 'Pressure Altitude (ft)', 'Turn Rate (deg/s)',
            'Lateral Accel (g)', 'Vertical Accel(g)', 'Angle of Attack (%)',
            'Vertical Speed (ft/min)', 'OAT (deg C)', 'True Airspeed (knots)',
            'Barometer Setting (inHg)', 'Density Altitude (ft)', 'Wind Direction (deg)',
            'Wind Speed (knots)', 'Heading Bug (deg)', 'Altitude Bug (ft)',
            'Airspeed Bug (knots)', 'Vertical Speed Bug (ft/min)', 'Course (deg)',
            'CDI Source Type', 'CDI Source Port', 'CDI Scale (NM)', 'CDI Deflection (%)',
            'Glideslope (%)', 'AP Engaged', 'AP Roll Mode', 'AP Roll Force',
            'AP Roll Position (steps)', 'AP Roll Slip (bool)', 'AP Pitch Force',
            'AP Pitch Position (steps)', 'AP Pitch Slip (bool)', 'AP Yaw Force',
            'AP Yaw Position', 'AP Yaw Slip (bool)', 'Transponder Status',
            'Transponder Reply (bool)', 'Transponder Identing (bool)', 'Transponder Code (octal)',
            'Oil Pressure (PSI)', 'Oil Temp (deg C)', 'RPM L', 'RPM R',
            'Manifold Pressure (inHg)', 'Fuel Flow 1 (gal/hr)', 'Fuel Flow 2 (gal/hr)',
            'Fuel Pressure (PSI)', 'Fuel Level L (gal)', 'Fuel Level R (gal)',
            'Fuel Remaining (gal)', 'Volts 1', 'Volts 2', 'Amps', 'Hobbs Time', 'Tach Time',
            'CHT 6 (deg C)', 'EGT 6 (deg C)', 'CHT 5 (deg C)', 'EGT 5 (deg C)',
            'CHT 4 (deg C)', 'EGT 4 (deg C)', 'CHT 3 (deg C)', 'EGT 3 (deg C)',
            'CHT 2 (deg C)', 'EGT 2 (deg C)', 'CHT 1 (deg C)', 'EGT 1 (deg C)',
            'Thermocouple 13 (deg C)', 'Thermocouple 14 (deg C)', 'LEFT CONTACT (V)',
            'GP Input 2', 'ALT CONTACT (V)', 'FUEL PRESSURE (PSI)', 'PHEAT CONTACT (V)',
            'CABIN TEMPERATURE (deg C)', 'FUEL CONTACT (V)', 'CANOPY CONTACT (V)',
            'LEFT LEVEL (gal)', 'RIGHT LEVEL (gal)', 'OIL PRESSURE (PSI)',
            'OIL TEMPERATURE (deg C)', 'GP Input 13', 'Contacts', 'Percent Power',
            'EGT Leaning State'
        ]
        
        # Create actual column names list based on the CSV structure
        column_names = []
        for i in range(num_columns):
            if i < len(base_columns):
                column_names.append(base_columns[i])
            else:
                column_names.append(f'Extra_Column_{i+1}')
        
        # Assign column names to the dataframe
        df.columns = column_names
        
        # Convert timestamp
        print("Processing timestamps...")
        df['timestamp'] = pd.to_datetime(df['GPS Date & Time'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        # Check for and handle duplicate timestamps
        print("Checking for duplicate timestamps...")
        duplicates = df.duplicated(subset=['timestamp'], keep=False)
        if duplicates.any():
            duplicate_count = duplicates.sum()
            print(f"Found {duplicate_count} rows with duplicate timestamps")
            
            # Add a microsecond offset to duplicates to make them unique
            # Group by timestamp and add incremental microseconds within each group
            grouped = df[duplicates].groupby('timestamp')
            
            for timestamp, group in grouped:
                # Skip the first row in each group (keep it as is)
                indices = group.index[1:]
                for i, idx in enumerate(indices):
                    # Add i+1 microseconds to make the timestamp unique
                    df.at[idx, 'timestamp'] = timestamp + pd.Timedelta(microseconds=i+1)
            
            # Verify uniqueness after modification
            if df.duplicated(subset=['timestamp']).any():
                print("Warning: Still have duplicate timestamps after adjustment")
            else:
                print("Successfully made all timestamps unique")
        
        if len(df) == 0:
            print("Error: No valid timestamped data found")
            return False
        
        # Get session time range
        start_time = df['timestamp'].min()
        end_time = df['timestamp'].max()
        print(f"Flight session spans from {start_time} to {end_time}")
        
        # Insert aircraft if it doesn't exist
        print(f"Ensuring aircraft {tail_number} exists in database...")
        with engine.connect() as conn:
            conn.execute(text(
                "INSERT INTO aircraft (tail_number) VALUES (:tail_number) ON CONFLICT DO NOTHING"
            ), {"tail_number": tail_number})
            conn.commit()
        
        # Create flight session
        print("Creating flight session record...")
        with engine.connect() as conn:
            result = conn.execute(text(
                """INSERT INTO flight_sessions (tail_number, start_time, end_time) 
                VALUES (:tail_number, :start_time, :end_time) 
                ON CONFLICT (tail_number, start_time) DO UPDATE
                SET end_time = EXCLUDED.end_time
                RETURNING session_id"""
            ), {"tail_number": tail_number, "start_time": start_time, "end_time": end_time})
            session_id = result.scalar()
            conn.commit()
            
        print(f"Session ID: {session_id}")
        
        # Prepare engine data
        print("Processing engine data...")
        engine_cols = ['timestamp']
        
        # Engine parameters
        engine_params = {
            'RPM L': 'rpm_left',
            'RPM R': 'rpm_right',
            'Oil Pressure (PSI)': 'oil_pressure',
            'Oil Temp (deg C)': 'oil_temp',
            'Manifold Pressure (inHg)': 'manifold_pressure'
        }
        
        # CHT/EGT parameters
        for i in range(1, 7):
            engine_params[f'CHT {i} (deg C)'] = f'cht_{i}'
            engine_params[f'EGT {i} (deg C)'] = f'egt_{i}'
        
        # Add columns that exist in the dataframe
        for col, db_col in engine_params.items():
            if col in df.columns:
                engine_cols.append(col)
        
        # Create engine dataframe
        if len(engine_cols) > 1:  # More than just timestamp
            engine_data = df[engine_cols].copy()
            engine_data['session_id'] = session_id
            
            # Rename columns to match database schema
            col_mapping = {key: val for key, val in engine_params.items() if key in engine_cols}
            engine_data = engine_data.rename(columns=col_mapping)
            
            # Convert to numeric, handling errors
            for col in engine_data.columns:
                if col not in ['timestamp', 'session_id']:
                    engine_data[col] = pd.to_numeric(engine_data[col], errors='coerce')
            
            # Drop rows with all NaN values for engine parameters
            numeric_cols = [col for col in engine_data.columns if col not in ['timestamp', 'session_id']]
            engine_data = engine_data.dropna(subset=numeric_cols, how='all')
            
            # Write to database in batches
            batch_size = 500
            total_rows = len(engine_data)
            print(f"Writing {total_rows} rows of engine data to database...")
            
            for i in range(0, total_rows, batch_size):
                end_idx = min(i + batch_size, total_rows)
                batch = engine_data.iloc[i:end_idx]
                
                try:
                    # Use to_sql with if_exists='append' to add records
                    batch.to_sql('engine_data', engine, if_exists='append', index=False)
                    print(f"Inserted batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")
                except Exception as e:
                    print(f"Error inserting batch: {e}")
                    print("Attempting to continue with next batch...")
            
            print(f"Completed engine data insertion")
        else:
            print("Warning: No engine columns found in CSV")
        
        # Record the file upload
        print("Recording file upload...")
        with engine.connect() as conn:
            conn.execute(text(
                """INSERT INTO file_uploads 
                (filename, status, processing_notes, tail_number, missions_added)
                VALUES (:filename, :status, :notes, :tail_number, :missions_added)"""
            ), {
                "filename": filename,
                "status": "completed",
                "notes": f"Processed {total_rows} data points",
                "tail_number": tail_number,
                "missions_added": 1
            })
            conn.commit()
        
        print(f"Successfully processed file: {filename}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Update this path to your CSV file location
    csv_file = '2023-08-11-SMK335-SN8889-15.2.0.4389-USER_LOG_DATA.csv'
    
    if os.path.exists(csv_file):
        process_csv(csv_file)
    else:
        print(f"File not found: {csv_file}")