import pandas as pd
import numpy as np
from pathlib import Path

class CanSatDataPreprocessor:
    """
    Modular preprocessor for CanSat flight data. 
    Handles whitespace stripping, unit conversions, and Geodetic to NED projection.
    """
    def __init__(self, filepath: str | Path):
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self._clean_column_names()

    def _clean_column_names(self):
        """Strips leading and trailing whitespace from all column names."""
        self.df.columns = self.df.columns.str.strip()

    def convert_geodetic_to_ned(self, lat_col: str, lon_col: str, alt_col: str = None):
        """
        Converts Geodetic Latitude and Longitude (degrees) to local North and East (meters).
        Uses a flat-earth approximation relative to the CanSat's initial coordinate.
        """
        if lat_col not in self.df.columns or lon_col not in self.df.columns:
            raise ValueError(f"Columns '{lat_col}' and '{lon_col}' not found.")
            
        R_EARTH = 6378137.0  # Earth radius in meters

        # The first GPS reading becomes the [0, 0] origin point for the EKF map
        lat0 = np.radians(self.df[lat_col].iloc[0])
        lon0 = np.radians(self.df[lon_col].iloc[0])

        lat_rad = np.radians(self.df[lat_col])
        lon_rad = np.radians(self.df[lon_col])

        # Calculate planar distance in meters along North and East axes
        self.df['pos_N'] = (lat_rad - lat0) * R_EARTH
        self.df['pos_E'] = (lon_rad - lon0) * R_EARTH * np.cos(lat0)
        
        if alt_col and alt_col in self.df.columns:
            self.df['altitude'] = self.df[alt_col]

    def map_gyro_data(self, gyro_yaw_col: str):
        """
        Extracts the yaw axis gyro reading and converts it to radians per second.
        Pads with zeros if the sensor data is missing from the CSV.
        """
        if gyro_yaw_col in self.df.columns:
            # Assuming GYRO_Y is logged in degrees/second; EKF math requires rad/s
            self.df['gyro_z'] = np.radians(self.df[gyro_yaw_col]) 
        else:
            # Fallback for datasets like navigation.csv that lack IMU data
            self.df['gyro_z'] = 0.0

    def get_ekf_dataframe(self) -> pd.DataFrame:
        """
        Returns only the specific columns required by the Extended Kalman Filter.
        """
        required_cols = ['pos_N', 'pos_E', 'altitude', 'gyro_z']
        
        # Ensure all columns exist to prevent KeyError in the EKF script
        for col in required_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0 
                
        return self.df[required_cols]