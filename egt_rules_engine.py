# egt_rules_engine.py
import pandas as pd
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text
import logging

logger = logging.getLogger(__name__)

class EGTRulesEngine:
    def __init__(self, db_engine):
        self.db_engine = db_engine
        self.initialize_alerts_table()
    
    def initialize_alerts_table(self):
        """Create EGT alerts table if it doesn't exist"""
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS egt_alerts (
                        id SERIAL PRIMARY KEY,
                        session_id INTEGER REFERENCES flight_sessions(session_id),
                        tail_number VARCHAR(20),
                        alert_timestamp TIMESTAMP NOT NULL,
                        alert_type VARCHAR(50) NOT NULL,
                        severity VARCHAR(20) NOT NULL,
                        title VARCHAR(200) NOT NULL,
                        description TEXT,
                        affected_cylinders TEXT,
                        egt_values JSONB,
                        egt_spread FLOAT,
                        deviation_amount FLOAT,
                        trend_direction VARCHAR(20),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        acknowledged BOOLEAN DEFAULT FALSE
                    )
                """))
                conn.commit()
                logger.info("EGT alerts table initialized")
        except Exception as e:
            logger.error(f"Error initializing EGT alerts table: {e}")
    
    def analyze_egt_data(self, session_id, tail_number, engine_data_df):
        """
        Analyze EGT data for anomalies and create alerts
        
        Args:
            session_id: Flight session ID
            tail_number: Aircraft tail number
            engine_data_df: DataFrame with engine data including EGT columns
        """
        try:
            logger.info(f"Analyzing EGT data for session {session_id}")
            
            # Extract EGT columns
            egt_columns = [col for col in engine_data_df.columns if 'egt_' in col.lower()]
            if len(egt_columns) < 2:
                logger.warning(f"Not enough EGT columns found for analysis: {egt_columns}")
                return
            
            # Clean and prepare data
            egt_data = engine_data_df[['timestamp'] + egt_columns].copy()
            egt_data = egt_data.dropna(subset=['timestamp'])
            
            # Convert EGT columns to numeric, replacing invalid values with NaN
            for col in egt_columns:
                egt_data[col] = pd.to_numeric(egt_data[col], errors='coerce')
            
            # Remove rows where all EGTs are NaN or zero
            valid_egt_mask = egt_data[egt_columns].apply(
                lambda row: (row > 0).any(), axis=1
            )
            egt_data = egt_data[valid_egt_mask].copy()
            
            if len(egt_data) < 10:  # Need at least 10 data points for trend analysis
                logger.warning(f"Not enough valid EGT data points for analysis: {len(egt_data)}")
                return
            
            # Calculate rolling statistics for trend analysis
            window_size = min(20, len(egt_data) // 5)  # Adaptive window size
            
            alerts = []
            
            # Rule 1: Detect diverging EGTs (spreading apart)
            divergence_alerts = self._detect_egt_divergence(egt_data, egt_columns, window_size)
            alerts.extend(divergence_alerts)
            
            # Rule 2: Detect individual cylinder anomalies (hot or cold)
            cylinder_alerts = self._detect_cylinder_anomalies(egt_data, egt_columns, window_size)
            alerts.extend(cylinder_alerts)
            
            # Rule 3: Detect rapid EGT changes
            rapid_change_alerts = self._detect_rapid_changes(egt_data, egt_columns)
            alerts.extend(rapid_change_alerts)
            
            # Rule 4: Detect sustained deviations
            sustained_alerts = self._detect_sustained_deviations(egt_data, egt_columns, window_size)
            alerts.extend(sustained_alerts)
            
            # Store alerts in database
            for alert in alerts:
                self._store_alert(session_id, tail_number, alert)
            
            logger.info(f"EGT analysis completed for session {session_id}. Found {len(alerts)} alerts.")
            
        except Exception as e:
            logger.error(f"Error analyzing EGT data for session {session_id}: {e}")
    
    def _detect_egt_divergence(self, egt_data, egt_columns, window_size):
        """Detect when EGTs are diverging (spreading apart)"""
        alerts = []
        
        # Calculate rolling EGT spread (max - min)
        egt_spread = egt_data[egt_columns].apply(
            lambda row: row.max() - row.min() if row.notna().sum() >= 2 else np.nan, 
            axis=1
        )
        
        # Calculate rolling average spread
        rolling_spread = egt_spread.rolling(window=window_size, min_periods=5).mean()
        
        # Detect increasing spread trend
        spread_trend = rolling_spread.diff().rolling(window=5).mean()
        
        # Find periods where spread is increasing significantly
        divergence_threshold = 5.0  # degrees per sample average
        high_spread_threshold = 80.0  # absolute spread threshold
        
        for i in range(len(egt_data)):
            current_spread = egt_spread.iloc[i]
            current_trend = spread_trend.iloc[i] if not pd.isna(spread_trend.iloc[i]) else 0
            
            # Check for divergence conditions
            if (current_spread > high_spread_threshold and current_trend > divergence_threshold):
                
                # Get EGT values at this timestamp
                egt_values = {}
                for col in egt_columns:
                    value = egt_data.iloc[i][col]
                    if not pd.isna(value) and value > 0:
                        cylinder_num = col.split('_')[1] if '_' in col else col[-1]
                        egt_values[f'egt_{cylinder_num}'] = float(value)
                
                alerts.append({
                    'timestamp': egt_data.iloc[i]['timestamp'],
                    'alert_type': 'egt_divergence',
                    'severity': 'HIGH' if current_spread > 100 else 'MEDIUM',
                    'title': f'EGT Divergence Detected - Spread: {current_spread:.1f}°C',
                    'description': f'EGT temperatures are diverging. Current spread: {current_spread:.1f}°C, '
                                 f'trend: +{current_trend:.1f}°C/sample. This may indicate uneven fuel distribution '
                                 f'or individual cylinder issues.',
                    'affected_cylinders': 'multiple',
                    'egt_values': egt_values,
                    'egt_spread': float(current_spread),
                    'deviation_amount': float(current_trend),
                    'trend_direction': 'increasing'
                })
        
        return alerts
    
    def _detect_cylinder_anomalies(self, egt_data, egt_columns, window_size):
        """Detect individual cylinders running hot or cold"""
        alerts = []
        
        # Calculate rolling averages for each cylinder
        rolling_egts = egt_data[egt_columns].rolling(window=window_size, min_periods=5).mean()
        
        for i in range(window_size, len(egt_data)):
            current_egts = rolling_egts.iloc[i]
            valid_egts = current_egts[current_egts > 0].dropna()
            
            if len(valid_egts) < 2:
                continue
            
            mean_egt = valid_egts.mean()
            std_egt = valid_egts.std()
            
            # Check each cylinder against the group average
            for col in egt_columns:
                if col in valid_egts.index:
                    cylinder_temp = valid_egts[col]
                    deviation = cylinder_temp - mean_egt
                    cylinder_num = col.split('_')[1] if '_' in col else col[-1]
                    
                    # Hot cylinder detection
                    if deviation > 50:  # 50°C above average
                        severity = 'CRITICAL' if deviation > 80 else 'HIGH'
                        alerts.append({
                            'timestamp': egt_data.iloc[i]['timestamp'],
                            'alert_type': 'hot_cylinder',
                            'severity': severity,
                            'title': f'Hot Cylinder #{cylinder_num} - {deviation:.1f}°C Above Average',
                            'description': f'Cylinder #{cylinder_num} is running {deviation:.1f}°C hotter than '
                                         f'average ({cylinder_temp:.1f}°C vs {mean_egt:.1f}°C average). '
                                         f'Check fuel injection and ignition for this cylinder.',
                            'affected_cylinders': f'cylinder_{cylinder_num}',
                            'egt_values': {f'egt_{cylinder_num}': float(cylinder_temp), 'average': float(mean_egt)},
                            'egt_spread': float(valid_egts.max() - valid_egts.min()),
                            'deviation_amount': float(deviation),
                            'trend_direction': 'hot'
                        })
                    
                    # Cold cylinder detection
                    elif deviation < -40:  # 40°C below average
                        severity = 'HIGH' if deviation < -60 else 'MEDIUM'
                        alerts.append({
                            'timestamp': egt_data.iloc[i]['timestamp'],
                            'alert_type': 'cold_cylinder',
                            'severity': severity,
                            'title': f'Cold Cylinder #{cylinder_num} - {abs(deviation):.1f}°C Below Average',
                            'description': f'Cylinder #{cylinder_num} is running {abs(deviation):.1f}°C cooler than '
                                         f'average ({cylinder_temp:.1f}°C vs {mean_egt:.1f}°C average). '
                                         f'Check fuel flow and ignition for this cylinder.',
                            'affected_cylinders': f'cylinder_{cylinder_num}',
                            'egt_values': {f'egt_{cylinder_num}': float(cylinder_temp), 'average': float(mean_egt)},
                            'egt_spread': float(valid_egts.max() - valid_egts.min()),
                            'deviation_amount': float(abs(deviation)),
                            'trend_direction': 'cold'
                        })
        
        return alerts
    
    def _detect_rapid_changes(self, egt_data, egt_columns):
        """Detect rapid EGT changes that might indicate instability"""
        alerts = []
        
        # Calculate rate of change for each cylinder
        for col in egt_columns:
            egt_series = egt_data[col].dropna()
            if len(egt_series) < 5:
                continue
                
            # Calculate differences between consecutive readings
            egt_diff = egt_series.diff().abs()
            
            # Find rapid changes (more than 30°C change in one reading)
            rapid_changes = egt_diff[egt_diff > 30]
            
            for idx, change in rapid_changes.items():
                cylinder_num = col.split('_')[1] if '_' in col else col[-1]
                current_temp = egt_series.loc[idx]
                
                severity = 'HIGH' if change > 50 else 'MEDIUM'
                
                alerts.append({
                    'timestamp': egt_data.loc[idx, 'timestamp'],
                    'alert_type': 'rapid_change',
                    'severity': severity,
                    'title': f'Rapid EGT Change - Cylinder #{cylinder_num}: {change:.1f}°C',
                    'description': f'Rapid temperature change detected in cylinder #{cylinder_num}. '
                                 f'Change: {change:.1f}°C in one reading (current: {current_temp:.1f}°C). '
                                 f'This may indicate engine instability or sensor issues.',
                    'affected_cylinders': f'cylinder_{cylinder_num}',
                    'egt_values': {f'egt_{cylinder_num}': float(current_temp)},
                    'egt_spread': None,
                    'deviation_amount': float(change),
                    'trend_direction': 'unstable'
                })
        
        return alerts
    
    def _detect_sustained_deviations(self, egt_data, egt_columns, window_size):
        """Detect sustained deviations from normal patterns"""
        alerts = []
        
        # Calculate overall baseline for the flight
        baseline_egts = {}
        for col in egt_columns:
            valid_temps = egt_data[col][egt_data[col] > 0].dropna()
            if len(valid_temps) > 10:
                # Use median of middle 50% as baseline (removes outliers)
                baseline_egts[col] = valid_temps.quantile([0.25, 0.75]).mean()
        
        if len(baseline_egts) < 2:
            return alerts
        
        # Look for sustained deviations from baseline
        for i in range(window_size, len(egt_data)):
            window_data = egt_data.iloc[i-window_size:i]
            
            for col in baseline_egts:
                cylinder_num = col.split('_')[1] if '_' in col else col[-1]
                baseline = baseline_egts[col]
                
                # Check if cylinder has been consistently above or below baseline
                recent_temps = window_data[col][window_data[col] > 0].dropna()
                
                if len(recent_temps) >= window_size * 0.7:  # At least 70% of window has data
                    avg_recent = recent_temps.mean()
                    deviation = avg_recent - baseline
                    
                    # Sustained high temperature
                    if deviation > 40 and (recent_temps > baseline + 30).all():
                        alerts.append({
                            'timestamp': egt_data.iloc[i]['timestamp'],
                            'alert_type': 'sustained_high',
                            'severity': 'HIGH' if deviation > 60 else 'MEDIUM',
                            'title': f'Sustained High EGT - Cylinder #{cylinder_num}',
                            'description': f'Cylinder #{cylinder_num} has been running {deviation:.1f}°C above '
                                         f'baseline for sustained period. Current: {avg_recent:.1f}°C, '
                                         f'baseline: {baseline:.1f}°C.',
                            'affected_cylinders': f'cylinder_{cylinder_num}',
                            'egt_values': {f'egt_{cylinder_num}': float(avg_recent), 'baseline': float(baseline)},
                            'egt_spread': None,
                            'deviation_amount': float(deviation),
                            'trend_direction': 'sustained_high'
                        })
                    
                    # Sustained low temperature
                    elif deviation < -30 and (recent_temps < baseline - 20).all():
                        alerts.append({
                            'timestamp': egt_data.iloc[i]['timestamp'],
                            'alert_type': 'sustained_low',
                            'severity': 'MEDIUM',
                            'title': f'Sustained Low EGT - Cylinder #{cylinder_num}',
                            'description': f'Cylinder #{cylinder_num} has been running {abs(deviation):.1f}°C below '
                                         f'baseline for sustained period. Current: {avg_recent:.1f}°C, '
                                         f'baseline: {baseline:.1f}°C.',
                            'affected_cylinders': f'cylinder_{cylinder_num}',
                            'egt_values': {f'egt_{cylinder_num}': float(avg_recent), 'baseline': float(baseline)},
                            'egt_spread': None,
                            'deviation_amount': float(abs(deviation)),
                            'trend_direction': 'sustained_low'
                        })
        
        return alerts
    
    def _store_alert(self, session_id, tail_number, alert):
        """Store alert in database"""
        try:
            with self.db_engine.connect() as conn:
                conn.execute(text("""
                    INSERT INTO egt_alerts 
                    (session_id, tail_number, alert_timestamp, alert_type, severity, 
                     title, description, affected_cylinders, egt_values, egt_spread, 
                     deviation_amount, trend_direction)
                    VALUES 
                    (:session_id, :tail_number, :timestamp, :alert_type, :severity,
                     :title, :description, :affected_cylinders, :egt_values, :egt_spread,
                     :deviation_amount, :trend_direction)
                """), {
                    'session_id': session_id,
                    'tail_number': tail_number,
                    'timestamp': alert['timestamp'],
                    'alert_type': alert['alert_type'],
                    'severity': alert['severity'],
                    'title': alert['title'],
                    'description': alert['description'],
                    'affected_cylinders': alert['affected_cylinders'],
                    'egt_values': alert['egt_values'],
                    'egt_spread': alert['egt_spread'],
                    'deviation_amount': alert['deviation_amount'],
                    'trend_direction': alert['trend_direction']
                })
                conn.commit()
        except Exception as e:
            logger.error(f"Error storing EGT alert: {e}")
    
    def get_session_alerts(self, session_id):
        """Get all EGT alerts for a session"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT * FROM egt_alerts 
                    WHERE session_id = :session_id 
                    ORDER BY alert_timestamp
                """), {'session_id': session_id})
                
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Error retrieving EGT alerts for session {session_id}: {e}")
            return []
    
    def get_aircraft_alert_summary(self, tail_number, days=30):
        """Get alert summary for an aircraft"""
        try:
            with self.db_engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_alerts,
                        COUNT(CASE WHEN severity = 'CRITICAL' THEN 1 END) as critical_count,
                        COUNT(CASE WHEN severity = 'HIGH' THEN 1 END) as high_count,
                        COUNT(CASE WHEN severity = 'MEDIUM' THEN 1 END) as medium_count,
                        alert_type,
                        COUNT(*) as type_count
                    FROM egt_alerts ea
                    JOIN flight_sessions fs ON ea.session_id = fs.session_id
                    WHERE ea.tail_number = :tail_number 
                    AND fs.start_time >= NOW() - INTERVAL '%s DAYS'
                    GROUP BY alert_type
                    ORDER BY type_count DESC
                """ % days), {'tail_number': tail_number})
                
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.error(f"Error retrieving EGT alert summary for {tail_number}: {e}")
            return []