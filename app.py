# app.py
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
import threading
from werkzeug.utils import secure_filename
from sqlalchemy import create_engine, text
import pandas as pd
import re
from datetime import datetime, timedelta
import numpy as np
import uuid
import logging

# Import the EGT Rules Engine
from egt_rules_engine import EGTRulesEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Database connection
db_user = "postgres"
db_password = "your_password_here"  # Replace with your actual password
db_host = "localhost"
db_port = "5432"
db_name = "aircraft_maintenance"

# Create connection string
db_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_string)

# Initialize EGT Rules Engine
egt_rules_engine = EGTRulesEngine(engine)

def extract_tail_number(filename):
    """Extract SMK tail number from filename"""
    match = re.search(r'SMK(\d+)', filename)
    if match:
        return f"SMK{match.group(1)}"
    else:
        return "UNKNOWN"  # Fallback value

def detect_missions(df, min_gap_minutes=30):
    """
    Detect multiple missions within a dataframe based on timestamp gaps.
    Returns a list of dataframes, one per mission.
    """
    if 'timestamp' not in df.columns or len(df) == 0:
        return [df]  # Return original dataframe if no timestamps
    
    # Sort by timestamp to ensure proper gap detection
    df = df.sort_values('timestamp')
    
    # Calculate time gaps between consecutive rows
    df['time_gap'] = df['timestamp'].diff()
    
    # Identify mission boundaries where gap >= specified minutes
    gap_threshold = timedelta(minutes=min_gap_minutes)
    mission_starts = [0]  # First row is always start of first mission
    
    for i in range(1, len(df)):
        if df.iloc[i]['time_gap'] >= gap_threshold:
            mission_starts.append(i)
    
    # Split dataframe into missions
    missions = []
    for i in range(len(mission_starts)):
        start_idx = mission_starts[i]
        end_idx = mission_starts[i+1] if i < len(mission_starts)-1 else len(df)
        mission_df = df.iloc[start_idx:end_idx].copy()
        
        # Generate a mission ID
        mission_start_time = mission_df['timestamp'].min()
        mission_id = f"M{mission_start_time.strftime('%Y%m%d%H%M')}"
        mission_df['mission_id'] = mission_id
        
        missions.append(mission_df)
    
    print(f"Detected {len(missions)} distinct missions in the data")
    
    # Print mission details for debugging
    for i, mission in enumerate(missions):
        start_time = mission['timestamp'].min()
        end_time = mission['timestamp'].max()
        duration = (end_time - start_time).total_seconds() / 60  # in minutes
        print(f"Mission {i+1}: {mission['mission_id'].iloc[0]} - Start: {start_time}, End: {end_time}, Duration: {duration:.1f} minutes, Rows: {len(mission)}")
    
    return missions

def process_mission(mission_df, tail_number, filename):
    """Process a single mission dataframe and store in database"""
    try:
        # Generate mission ID if not already present
        if 'mission_id' not in mission_df.columns:
            mission_start_time = mission_df['timestamp'].min()
            mission_id = f"M{mission_start_time.strftime('%Y%m%d%H%M')}"
            mission_df['mission_id'] = mission_id
        else:
            mission_id = mission_df['mission_id'].iloc[0]
        
        # Get session time range
        start_time = mission_df['timestamp'].min()
        end_time = mission_df['timestamp'].max()
        
        print(f"Processing mission {mission_id} from {start_time} to {end_time}")
        
        # Check if this mission already exists in the database
        with engine.connect() as conn:
            existing = conn.execute(text("""
                SELECT session_id FROM flight_sessions 
                WHERE tail_number = :tail_number 
                AND ABS(EXTRACT(EPOCH FROM (start_time - :start_time))) < 60  -- Within 1 minute
                AND mission_id = :mission_id
            """), {
                "tail_number": tail_number,
                "start_time": start_time,
                "mission_id": mission_id
            }).fetchone()
            
            if existing:
                print(f"Mission {mission_id} already exists in database, skipping")
                # Return the existing session_id for reference
                return {"success": True, "session_id": existing[0], "new": False}
        
        # Create flight session for this mission
        with engine.connect() as conn:
            result = conn.execute(text(
                """INSERT INTO flight_sessions (tail_number, start_time, end_time, mission_id) 
                VALUES (:tail_number, :start_time, :end_time, :mission_id)
                RETURNING session_id"""
            ), {
                "tail_number": tail_number, 
                "start_time": start_time, 
                "end_time": end_time,
                "mission_id": mission_id
            })
            session_id = result.scalar()
            conn.commit()
            
        print(f"Created session ID: {session_id} for mission {mission_id}")
        
        # Prepare engine data
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
            if col in mission_df.columns:
                engine_cols.append(col)
        
        # Create engine dataframe
        if len(engine_cols) > 1:  # More than just timestamp
            engine_data = mission_df[engine_cols].copy()
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
            print(f"Writing {total_rows} rows of engine data to database for mission {mission_id}...")
            
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
            
            print(f"Completed engine data insertion for mission {mission_id}")
            
            # Analyze EGT data for anomalies using rules engine
            try:
                print(f"Analyzing EGT data for mission {mission_id}...")
                egt_rules_engine.analyze_egt_data(session_id, tail_number, engine_data)
                print(f"EGT analysis completed for mission {mission_id}")
            except Exception as e:
                print(f"Error in EGT rules analysis for mission {mission_id}: {e}")
            
        else:
            print(f"Warning: No engine columns found for mission {mission_id}")
        
        # Process flight data (position, attitude, etc.)
        flight_cols = ['timestamp']
        flight_params = {
            'Latitude (deg)': 'latitude',
            'Longitude (deg)': 'longitude',
            'GPS Altitude (feet)': 'altitude',
            'Ground Speed (knots)': 'ground_speed',
            'Pitch (deg)': 'pitch',
            'Roll (deg)': 'roll',
            'Magnetic Heading (deg)': 'magnetic_heading',
            'Vertical Speed (ft/min)': 'vertical_speed',
            'Indicated Airspeed (knots)': 'indicated_airspeed',
            'True Airspeed (knots)': 'true_airspeed'
        }
        
        # Add columns that exist in the dataframe
        for col, db_col in flight_params.items():
            if col in mission_df.columns:
                flight_cols.append(col)
        
        # Create flight dataframe if we have flight data
        if len(flight_cols) > 1:
            flight_data = mission_df[flight_cols].copy()
            flight_data['session_id'] = session_id
            
            # Rename columns to match database schema
            col_mapping = {key: val for key, val in flight_params.items() if key in flight_cols}
            flight_data = flight_data.rename(columns=col_mapping)
            
            # Convert to numeric, handling errors
            for col in flight_data.columns:
                if col not in ['timestamp', 'session_id']:
                    flight_data[col] = pd.to_numeric(flight_data[col], errors='coerce')
            
            # Write to database in batches
            batch_size = 500
            total_rows = len(flight_data)
            print(f"Writing {total_rows} rows of flight data to database for mission {mission_id}...")
            
            for i in range(0, total_rows, batch_size):
                end_idx = min(i + batch_size, total_rows)
                batch = flight_data.iloc[i:end_idx]
                
                try:
                    batch.to_sql('flight_data', engine, if_exists='append', index=False)
                    print(f"Inserted flight data batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")
                except Exception as e:
                    print(f"Error inserting flight data batch: {e}")
        
        # Process fuel system data
        fuel_cols = ['timestamp']
        fuel_params = {
            'Fuel Flow 1 (gal/hr)': 'fuel_flow_1',
            'Fuel Flow 2 (gal/hr)': 'fuel_flow_2',
            'Fuel Level L (gal)': 'fuel_level_left',
            'Fuel Level R (gal)': 'fuel_level_right',
            'Fuel Pressure (PSI)': 'fuel_pressure',
            'Fuel Remaining (gal)': 'fuel_remaining'
        }
        
        # Add columns that exist in the dataframe
        for col, db_col in fuel_params.items():
            if col in mission_df.columns:
                fuel_cols.append(col)
        
        # Create fuel dataframe if we have fuel data
        if len(fuel_cols) > 1:
            fuel_data = mission_df[fuel_cols].copy()
            fuel_data['session_id'] = session_id
            
            # Rename columns to match database schema
            col_mapping = {key: val for key, val in fuel_params.items() if key in fuel_cols}
            fuel_data = fuel_data.rename(columns=col_mapping)
            
            # Convert to numeric, handling errors
            for col in fuel_data.columns:
                if col not in ['timestamp', 'session_id']:
                    fuel_data[col] = pd.to_numeric(fuel_data[col], errors='coerce')
            
            # Write to database in batches
            batch_size = 500
            total_rows = len(fuel_data)
            print(f"Writing {total_rows} rows of fuel data to database for mission {mission_id}...")
            
            for i in range(0, total_rows, batch_size):
                end_idx = min(i + batch_size, total_rows)
                batch = fuel_data.iloc[i:end_idx]
                
                try:
                    batch.to_sql('fuel_system', engine, if_exists='append', index=False)
                    print(f"Inserted fuel data batch {i//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")
                except Exception as e:
                    print(f"Error inserting fuel data batch: {e}")
        
        return {"success": True, "session_id": session_id, "new": True}
        
    except Exception as e:
        print(f"Error processing mission: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def process_csv(filepath, skip_rows=3):
    """Process a single CSV file with multiple missions"""
    try:
        # Get filename and extract tail number
        filename = os.path.basename(filepath)
        tail_number = extract_tail_number(filename)
        print(f"Processing file: {filename}")
        print(f"Extracted tail number: {tail_number}")
        
        # Insert aircraft if it doesn't exist
        print(f"Ensuring aircraft {tail_number} exists in database...")
        with engine.connect() as conn:
            conn.execute(text(
                "INSERT INTO aircraft (tail_number) VALUES (:tail_number) ON CONFLICT DO NOTHING"
            ), {"tail_number": tail_number})
            conn.commit()
        
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
        
        # Detect multiple missions in the file
        missions = detect_missions(df, min_gap_minutes=30)
        
        # Track the processed missions
        processed_missions = []
        new_missions = 0
        
        # Process each mission
        for mission_df in missions:
            mission_result = process_mission(mission_df, tail_number, filename)
            
            if mission_result["success"]:
                processed_missions.append(mission_result["session_id"])
                if mission_result["new"]:
                    new_missions += 1
        
        # Record the file upload
        notes = f"Processed {len(missions)} missions, {new_missions} new"
        print(notes)
        
        with engine.connect() as conn:
            conn.execute(text(
                """INSERT INTO file_uploads 
                (filename, status, processing_notes, tail_number, missions_added)
                VALUES (:filename, :status, :notes, :tail_number, :missions_added)"""
            ), {
                "filename": filename,
                "status": "completed",
                "notes": notes,
                "tail_number": tail_number,
                "missions_added": new_missions
            })
            conn.commit()
        
        print(f"Successfully processed file: {filename}")
        return True
        
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False

@app.route('/')
def index():
    # Get list of aircraft with mission counts
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT a.tail_number, 
                   COUNT(fs.session_id) AS mission_count,
                   MAX(fs.end_time) AS last_mission
            FROM aircraft a
            LEFT JOIN flight_sessions fs ON a.tail_number = fs.tail_number
            GROUP BY a.tail_number
            ORDER BY a.tail_number
        """))
        # Fix: Use row._mapping to convert the result to a dictionary
        aircraft_list = [dict(row._mapping) for row in result]
    
    return render_template('index.html', aircraft=aircraft_list)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file part exists
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
            
        if file and file.filename.endswith('.csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process file
            result = process_csv(filepath)
            if result:
                flash(f'File processed successfully. EGT analysis completed.')
            else:
                flash('Error processing file')
            
            return redirect(url_for('index'))
        
        flash('File must be a CSV')
        return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/aircraft/<tail_number>')
def aircraft_detail(tail_number):
    # Get aircraft details
    with engine.connect() as conn:
        # Get basic info
        aircraft_info = conn.execute(text("""
            SELECT * FROM aircraft WHERE tail_number = :tail
        """), {"tail": tail_number}).fetchone()
        
        if not aircraft_info:
            flash(f"Aircraft {tail_number} not found")
            return redirect(url_for('index'))
        
        # Fix: Convert to dictionary using _mapping
        aircraft_info = dict(aircraft_info._mapping) if aircraft_info else None
        
        # Get mission list
        missions = conn.execute(text("""
            SELECT session_id, start_time, end_time, mission_id,
                   EXTRACT(EPOCH FROM (end_time - start_time))/60 AS duration_minutes
            FROM flight_sessions
            WHERE tail_number = :tail
            ORDER BY start_time DESC
        """), {"tail": tail_number})
        
        # Fix: Convert to dictionary using _mapping
        mission_list = [dict(row._mapping) for row in missions]
    
    # Get EGT alert summary for this aircraft
    alert_summary = egt_rules_engine.get_aircraft_alert_summary(tail_number, days=30)
    
    return render_template('aircraft_detail.html', 
                          aircraft=aircraft_info, 
                          missions=mission_list,
                          alert_summary=alert_summary)

@app.route('/mission/<int:session_id>')
def mission_detail(session_id):
    # Get mission details
    with engine.connect() as conn:
        mission_info = conn.execute(text("""
            SELECT fs.*, a.tail_number 
            FROM flight_sessions fs
            JOIN aircraft a ON fs.tail_number = a.tail_number
            WHERE fs.session_id = :id
        """), {"id": session_id}).fetchone()
        
        if not mission_info:
            flash(f"Mission ID {session_id} not found")
            return redirect(url_for('index'))
        
        # Fix: Convert to dictionary using _mapping
        mission_info = dict(mission_info._mapping) if mission_info else None
        
        # Get mission metrics
        metrics_row = conn.execute(text("""
            SELECT 
                COUNT(*) AS data_points,
                MAX(rpm_left) AS max_rpm_left,
                MAX(rpm_right) AS max_rpm_right,
                MAX(oil_temp) AS max_oil_temp,
                MIN(oil_pressure) AS min_oil_pressure,
                MAX(GREATEST(COALESCE(cht_1, 0), COALESCE(cht_2, 0), COALESCE(cht_3, 0), 
                           COALESCE(cht_4, 0), COALESCE(cht_5, 0), COALESCE(cht_6, 0))) AS max_cht
            FROM engine_data
            WHERE session_id = :id
        """), {"id": session_id}).fetchone()
        
        # Fix: Convert to dictionary using _mapping
        metrics = dict(metrics_row._mapping) if metrics_row else {}
        
        # Get EGT alerts for this session
        egt_alerts = egt_rules_engine.get_session_alerts(session_id)
        
        # Categorize alerts by severity
        alert_summary = {
            'total': len(egt_alerts),
            'critical': len([a for a in egt_alerts if a['severity'] == 'CRITICAL']),
            'high': len([a for a in egt_alerts if a['severity'] == 'HIGH']),
            'medium': len([a for a in egt_alerts if a['severity'] == 'MEDIUM']),
            'low': len([a for a in egt_alerts if a['severity'] == 'LOW'])
        }
    
    # Grafana URLs for embedded dashboards
    grafana_host = "http://localhost:3000"  # Change as needed for your environment
    grafana_urls = {
        'engine': f"http://localhost:3000/public-dashboards/781b2e6c92e449a583ecfee71048f1d8?var-session_id={session_id}",
        'flight': f"http://localhost:3000/d/flight-performance/flight-performance?orgId=1&var-session_id={session_id}&kiosk=tv",
        'system': f"http://localhost:3000/d/system-status/system-status?orgId=1&var-session_id={session_id}&kiosk=tv",
        'anomaly': f"http://localhost:3000/d/anomaly-detection/anomaly-detection?orgId=1&var-session_id={session_id}&kiosk=tv",
        'egt': f"http://localhost:3000/d/egt-analysis/egt-analysis?orgId=1&var-session_id={session_id}&kiosk=tv"
    }
    
    return render_template('mission_detail.html',
                          mission=mission_info,
                          metrics=metrics,
                          grafana_urls=grafana_urls,
                          egt_alerts=egt_alerts,
                          alert_summary=alert_summary)

def analyze_maintenance_needs(tail_number):
    """
    Analyze maintenance needs based on trend analysis including EGT anomalies
    Returns recommendations as a list of dictionaries
    """
    with engine.connect() as conn:
        # Get trend data
        result = conn.execute(text("""
            WITH time_periods AS (
                SELECT 
                    session_id,
                    tail_number,
                    start_time,
                    end_time,
                    RANK() OVER (PARTITION BY tail_number ORDER BY start_time DESC) as recency_rank
                FROM flight_sessions
                WHERE tail_number = :tail
                ORDER BY start_time DESC
                LIMIT 10
            ),
            engine_metrics AS (
                SELECT 
                    fs.session_id,
                    fs.start_time,
                    MAX(GREATEST(COALESCE(ed.cht_1, 0), COALESCE(ed.cht_2, 0), COALESCE(ed.cht_3, 0), 
                               COALESCE(ed.cht_4, 0), COALESCE(ed.cht_5, 0), COALESCE(ed.cht_6, 0))) as max_cht,
                    AVG((COALESCE(ed.cht_1, 0) + COALESCE(ed.cht_2, 0) + COALESCE(ed.cht_3, 0) + 
                         COALESCE(ed.cht_4, 0) + COALESCE(ed.cht_5, 0) + COALESCE(ed.cht_6, 0))/
                         NULLIF(CASE WHEN ed.cht_1 IS NOT NULL THEN 1 ELSE 0 END +
                                CASE WHEN ed.cht_2 IS NOT NULL THEN 1 ELSE 0 END +
                                CASE WHEN ed.cht_3 IS NOT NULL THEN 1 ELSE 0 END +
                                CASE WHEN ed.cht_4 IS NOT NULL THEN 1 ELSE 0 END +
                                CASE WHEN ed.cht_5 IS NOT NULL THEN 1 ELSE 0 END +
                                CASE WHEN ed.cht_6 IS NOT NULL THEN 1 ELSE 0 END, 0)) as avg_cht,
                    MIN(ed.oil_pressure) as min_oil_pressure,
                    MAX(ed.oil_temp) as max_oil_temp,
                    MAX(ed.rpm_left) as max_rpm_left
                FROM engine_data ed
                JOIN time_periods fs ON ed.session_id = fs.session_id
                GROUP BY fs.session_id, fs.start_time
                ORDER BY fs.start_time DESC
            )
            SELECT
                session_id,
                start_time,
                max_cht,
                avg_cht,
                min_oil_pressure,
                max_oil_temp,
                max_rpm_left,
                -- Calculate trends (difference from session to next)
                max_cht - LAG(max_cht) OVER (ORDER BY start_time) as cht_trend,
                min_oil_pressure - LAG(min_oil_pressure) OVER (ORDER BY start_time) as oil_pressure_trend
            FROM engine_metrics
            ORDER BY start_time DESC
        """), {"tail": tail_number})
        
        # Fix: Convert to dictionary using _mapping
        trend_data = [dict(row._mapping) for row in result]
        recommendations = []
    
    # Analyze CHT trends
    cht_trends = [row.get('cht_trend') for row in trend_data if row.get('cht_trend') is not None]
    if cht_trends and len(cht_trends) >= 3:
        # If we have consecutive increasing CHT trends
        if all(trend > 0 for trend in cht_trends[:3]):
            recommendations.append({
                'component': 'Cooling System',
                'severity': 'Warning',
                'message': 'Increasing CHT trends detected across multiple flights. Recommend cooling system inspection.',
                'trend_value': sum(cht_trends[:3])
            })
    
    # Analyze oil pressure trends
    oil_trends = [row.get('oil_pressure_trend') for row in trend_data if row.get('oil_pressure_trend') is not None]
    if oil_trends and len(oil_trends) >= 3:
        # If we have consecutive decreasing oil pressure
        if all(trend < 0 for trend in oil_trends[:3]):
            recommendations.append({
                'component': 'Oil System',
                'severity': 'Alert',
                'message': 'Decreasing oil pressure trend detected. Recommend immediate oil system inspection.',
                'trend_value': sum(oil_trends[:3])
            })
    
    # Check absolute values
    latest = trend_data[0] if trend_data else {}
    if latest.get('max_cht', 0) > 380:
        recommendations.append({
            'component': 'Cylinder Head',
            'severity': 'Alert',
            'message': f'Excessive cylinder head temperature detected ({latest.get("max_cht")}°C). Inspect cooling system and CHT sensor.',
            'trend_value': None
        })
    
    if latest.get('min_oil_pressure', 100) < 35:
        recommendations.append({
            'component': 'Oil System',
            'severity': 'Critical',
            'message': f'Low oil pressure detected ({latest.get("min_oil_pressure")} PSI). Immediate maintenance required.',
            'trend_value': None
        })
    
    # Add EGT-based recommendations from rules engine
    try:
        egt_summary = egt_rules_engine.get_aircraft_alert_summary(tail_number, days=30)
        
        for summary in egt_summary:
            alert_type = summary.get('alert_type', '')
            type_count = summary.get('type_count', 0)
            
            if type_count > 0:
                if alert_type == 'hot_cylinder':
                    recommendations.append({
                        'component': 'EGT - Hot Cylinder',
                        'severity': 'Critical' if type_count > 5 else 'Alert',
                        'message': f'Hot cylinder alerts detected: {type_count} incidents. Immediate cylinder inspection required.',
                        'trend_value': type_count
                    })
                elif alert_type == 'egt_divergence':
                    recommendations.append({
                        'component': 'EGT - Divergence Pattern',
                        'severity': 'Alert' if type_count > 3 else 'Warning',
                        'message': f'EGT divergence pattern detected {type_count} times. Check fuel distribution and cylinder balance.',
                        'trend_value': type_count
                    })
                elif alert_type == 'cold_cylinder':
                    recommendations.append({
                        'component': 'EGT - Cold Cylinder',
                        'severity': 'Warning',
                        'message': f'Cold cylinder pattern detected {type_count} times. Check fuel flow and ignition systems.',
                        'trend_value': type_count
                    })
                elif alert_type == 'rapid_change':
                    recommendations.append({
                        'component': 'EGT - Instability',
                        'severity': 'Alert',
                        'message': f'Rapid EGT changes detected {type_count} times. Check engine control systems.',
                        'trend_value': type_count
                    })
                elif alert_type == 'sustained_high':
                    recommendations.append({
                        'component': 'EGT - Sustained High Temps',
                        'severity': 'Alert',
                        'message': f'Sustained high temperatures detected {type_count} times. Monitor for engine wear.',
                        'trend_value': type_count
                    })
    except Exception as e:
        print(f"Error getting EGT recommendations for {tail_number}: {e}")
    
    return recommendations

@app.route('/api/maintenance/<tail_number>')
def get_maintenance_recommendations(tail_number):
    recommendations = analyze_maintenance_needs(tail_number)
    return jsonify(recommendations)

@app.route('/maintenance/<tail_number>')
def maintenance_view(tail_number):
    # Get aircraft details
    with engine.connect() as conn:
        aircraft_info = conn.execute(text("""
            SELECT * FROM aircraft WHERE tail_number = :tail
        """), {"tail": tail_number}).fetchone()
        
        if not aircraft_info:
            flash(f"Aircraft {tail_number} not found")
            return redirect(url_for('index'))
        
        # Fix: Convert to dictionary using _mapping
        aircraft_info = dict(aircraft_info._mapping) if aircraft_info else None
    
    # Get maintenance recommendations
    recommendations = analyze_maintenance_needs(tail_number)
    
    return render_template('maintenance.html', 
                          aircraft=aircraft_info,
                          recommendations=recommendations)

# EGT Alert API Endpoints
@app.route('/api/egt_alerts/<int:session_id>')
def get_egt_alerts(session_id):
    """API endpoint to get EGT alerts for a session"""
    try:
        alerts = egt_rules_engine.get_session_alerts(session_id)
        return jsonify({
            'session_id': session_id,
            'alert_count': len(alerts),
            'alerts': alerts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/egt_summary/<tail_number>')
def get_egt_alert_summary(tail_number):
    """API endpoint to get EGT alert summary for an aircraft"""
    try:
        days = request.args.get('days', 30, type=int)
        summary = egt_rules_engine.get_aircraft_alert_summary(tail_number, days)
        return jsonify({
            'tail_number': tail_number,
            'days': days,
            'summary': summary
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/egt_alerts/<int:session_id>')
def egt_alerts_detail(session_id):
    """Detailed view of EGT alerts for a mission"""
    # Get session details
    with engine.connect() as conn:
        mission_info = conn.execute(text("""
            SELECT fs.*, a.tail_number 
            FROM flight_sessions fs
            JOIN aircraft a ON fs.tail_number = a.tail_number
            WHERE fs.session_id = :id
        """), {"id": session_id}).fetchone()
        
        if not mission_info:
            flash(f"Mission ID {session_id} not found")
            return redirect(url_for('index'))
        
        mission_info = dict(mission_info._mapping) if mission_info else None
    
    # Get EGT alerts
    egt_alerts = egt_rules_engine.get_session_alerts(session_id)
    
    # Group alerts by type for summary
    alerts_by_type = {}
    for alert in egt_alerts:
        alert_type = alert['alert_type']
        if alert_type not in alerts_by_type:
            alerts_by_type[alert_type] = []
        alerts_by_type[alert_type].append(alert)
    
    return render_template('egt_alerts_detail.html',
                          mission=mission_info,
                          egt_alerts=egt_alerts,
                          alerts_by_type=alerts_by_type)

# Legacy anomaly detection API (Basic statistical analysis)
@app.route('/api/anomaly_detection/<int:session_id>')
def anomaly_detection_api(session_id):
    """API for basic statistical anomaly detection for a mission"""
    try:
        # Get engine data for the session
        engine_data = pd.read_sql(
            "SELECT timestamp, rpm_left, oil_pressure, oil_temp, manifold_pressure, "
            "cht_1, cht_2, cht_3, cht_4, cht_5, cht_6, "
            "egt_1, egt_2, egt_3, egt_4, egt_5, egt_6 "
            "FROM engine_data WHERE session_id = %s ORDER BY timestamp",
            engine, params=[session_id]
        )
        
        if len(engine_data) == 0:
            return jsonify({"error": "No data found for this session"}), 404
        
        # Basic statistical anomaly detection
        # 1. Z-score calculation for key parameters
        anomalies = []
        
        # Check for CHT anomalies
        for i in range(1, 7):
            cht_col = f'cht_{i}'
            if cht_col in engine_data.columns:
                # Calculate mean and standard deviation
                mean_val = engine_data[cht_col].mean()
                std_val = engine_data[cht_col].std()
                
                if std_val > 0:  # Avoid division by zero
                    # Calculate z-scores
                    z_scores = (engine_data[cht_col] - mean_val) / std_val
                    
                    # Find anomalies (z-score > 3)
                    anomaly_indices = engine_data.index[abs(z_scores) > 3].tolist()
                    
                    for idx in anomaly_indices:
                        timestamp = engine_data.iloc[idx]['timestamp']
                        value = engine_data.iloc[idx][cht_col]
                        anomalies.append({
                            "timestamp": timestamp.isoformat(),
                            "parameter": f"CHT {i}",
                            "value": float(value),
                            "mean": float(mean_val),
                            "z_score": float(z_scores[idx]),
                            "threshold": 3
                        })
        
        # Check for oil pressure anomalies
        if 'oil_pressure' in engine_data.columns:
            mean_val = engine_data['oil_pressure'].mean()
            std_val = engine_data['oil_pressure'].std()
            
            if std_val > 0:
                z_scores = (engine_data['oil_pressure'] - mean_val) / std_val
                anomaly_indices = engine_data.index[abs(z_scores) > 3].tolist()
                
                for idx in anomaly_indices:
                    timestamp = engine_data.iloc[idx]['timestamp']
                    value = engine_data.iloc[idx]['oil_pressure']
                    anomalies.append({
                        "timestamp": timestamp.isoformat(),
                        "parameter": "Oil Pressure",
                        "value": float(value),
                        "mean": float(mean_val),
                        "z_score": float(z_scores[idx]),
                        "threshold": 3
                    })
        
        # Sort anomalies by timestamp
        anomalies.sort(key=lambda x: x["timestamp"])
        
        # Return results
        return jsonify({
            "session_id": session_id,
            "anomaly_count": len(anomalies),
            "anomalies": anomalies
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/anomaly_detection/<int:session_id>')
def anomaly_detection_view(session_id):
    """View for basic anomaly detection results"""
    # Get session details
    with engine.connect() as conn:
        mission_info = conn.execute(text("""
            SELECT fs.*, a.tail_number 
            FROM flight_sessions fs
            JOIN aircraft a ON fs.tail_number = a.tail_number
            WHERE fs.session_id = :id
        """), {"id": session_id}).fetchone()
        
        if not mission_info:
            flash(f"Mission ID {session_id} not found")
            return redirect(url_for('index'))
        
        mission_info = dict(mission_info._mapping) if mission_info else None
    
    # Get anomaly detection results via the API
    try:
        import requests
        api_url = f"http://localhost:5000/api/anomaly_detection/{session_id}"
        response = requests.get(api_url)
        anomaly_data = response.json()
    except Exception as e:
        anomaly_data = {"error": str(e), "anomalies": []}
    
    return render_template('anomaly_detection.html',
                          mission=mission_info,
                          anomaly_data=anomaly_data)

# Additional utility routes
@app.route('/test_egt_analysis/<int:session_id>')
def test_egt_analysis(session_id):
    """Test EGT analysis on a specific session"""
    try:
        # Get engine data for this session
        with engine.connect() as conn:
            # Get session info
            session_info = conn.execute(text("""
                SELECT fs.*, a.tail_number 
                FROM flight_sessions fs
                JOIN aircraft a ON fs.tail_number = a.tail_number
                WHERE fs.session_id = :id
            """), {"id": session_id}).fetchone()
            
            if not session_info:
                flash(f"Session {session_id} not found")
                return redirect(url_for('index'))
            
            session_info = dict(session_info._mapping)
            
            # Get engine data
            engine_data_df = pd.read_sql("""
                SELECT timestamp, egt_1, egt_2, egt_3, egt_4, egt_5, egt_6,
                       cht_1, cht_2, cht_3, cht_4, cht_5, cht_6,
                       oil_pressure, oil_temp, rpm_left, rpm_right
                FROM engine_data 
                WHERE session_id = %s
                ORDER BY timestamp
            """, engine, params=[session_id])
            
            if len(engine_data_df) == 0:
                flash(f"No engine data found for session {session_id}")
                return redirect(url_for('mission_detail', session_id=session_id))
            
            # Run EGT analysis
            print(f"Testing EGT analysis on session {session_id} with {len(engine_data_df)} data points")
            egt_rules_engine.analyze_egt_data(session_id, session_info['tail_number'], engine_data_df)
            
            flash(f"EGT analysis completed for session {session_id}. Check the EGT alerts tab.")
            return redirect(url_for('mission_detail', session_id=session_id))
            
    except Exception as e:
        flash(f"Error during EGT analysis: {str(e)}")
        return redirect(url_for('index'))

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        # Test EGT rules engine
        egt_rules_available = egt_rules_engine is not None
        
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'egt_rules_engine': 'available' if egt_rules_available else 'unavailable',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    print("Starting Aircraft Predictive Maintenance System...")
    print("EGT Rules Engine initialized and ready")
    print("Access the application at: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)