import matplotlib.pyplot as plt # Plot data
import numpy as np    # Math features
import pandas as pd   # Manipulate Data
from os import walk   # Show files inside folder
from scipy import fftpack # FFT
from scipy.signal import butter, lfilter # Filter signal


class ProcessVibrationData:

    def __init__(self, path2data):
        # Folder with data (e.g.: './data/imbalance/imbalance/20g/')
        self.path2data = path2data

        # Global parameters from MAFAULDA datasheet
        self.acc_sample_rate = 50000.0
        # Accelerometer 1 - IMI Sensors, Model 601A01
        self.acc1_max_freq_range = 10000.0
        self.acc1_min_freq_range = 0.27
        # Accelerometer 2 - IMI Sensors, Model 604B31
        self.acc2_max_freq_range = 5000.0
        self.acc2_min_freq_range = 0.5

    def get_all_filenames(self):
        # Get .csv files from folder
        # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        filenames = next(walk(self.path2data), (None, None, []))[2]  
        return filenames
    
    def open_df(self, filename_index = -1):
        # Get .csv files from folder
        # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
        filenames = next(walk(self.path2data), (None, None, []))[2]  
        # Get dataframe
        self.df = pd.read_csv(self.path2data + filenames[filename_index])

    def rename_df_columns(self):
        self.df = self.df.rename(columns={ self.df.columns[0]: "tachometer" }) # tachometer signal
        self.df = self.df.rename(columns={ self.df.columns[1]: "under_bear_acc1_axial" }) # underhang bearing accelerometer axial
        self.df = self.df.rename(columns={ self.df.columns[2]: "under_bear_acc1_radial" }) # underhang bearing accelerometer radial
        self.df = self.df.rename(columns={ self.df.columns[3]: "under_bear_acc1_tang" }) # underhang bearing accelerometer tangential
        self.df = self.df.rename(columns={ self.df.columns[4]: "over_bear_acc2_axial" }) # overhang bearing accelerometer axial
        self.df = self.df.rename(columns={ self.df.columns[5]: "over_bear_acc2_radial" }) # overhang bearing accelerometer radial
        self.df = self.df.rename(columns={ self.df.columns[6]: "over_bear_acc2_tang" }) # overhang bearing accelerometer tangential
        self.df = self.df.rename(columns={ self.df.columns[7]: "microphone" }) # microphone

    def get_tachometer_freq(self):
        # Calculate totam measurements per seconds
        total_measurement_sec = float(self.df.shape[0]/self.acc_sample_rate)
        # Define Limits 
        tach_sig_tresh_min = 2.0
        tach_sig_tresh_max = 5.1
        # Create new column with True when tachometer output is positive
        self.df['df_between'] = self.df['tachometer'].between(tach_sig_tresh_min, tach_sig_tresh_max)
        # New column with previous shifted
        self.df['df_between_shift'] = self.df['df_between'].shift(-1)
        # Logic test, when before is False and actual is True
        self.df['df_between_logic'] = np.where((self.df['df_between'] == True) & (self.df['df_between_shift'] == False), True, False)
        # Calculate average output
        tachometer_freq = int(self.df[self.df['df_between_logic'] == True].shape[0]/total_measurement_sec)
        return tachometer_freq

    def get_sel_fft_df(self, tachometer_freq, start_data=0):
        # FFT parameters
        points_rev = int(self.acc_sample_rate/tachometer_freq)
        # Get 5 times motor revolution data
        n_rev_points = points_rev * 5
        df_fft = self.df.iloc[start_data:start_data + n_rev_points]
        return df_fft

    def lowpass_filter_array(self, array_fft, sensor_max_freq_range):
        # Filter signal - Low pass filter
        order = 5
        cutoff = sensor_max_freq_range
        fs = self.acc_sample_rate
        b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
        array_fft_filt = lfilter(b, a, array_fft)
        return array_fft_filt

    def lowpass_filter_df(self, df_fft):
        # Create dataframe 
        df_fft_filt = pd.DataFrame()
        # Filter Accelerometer 1
        column_str = "under_bear_acc1_axial"
        df_fft_filt[column_str] =  self.lowpass_filter_array(np.array(df_fft[column_str]), self.acc1_max_freq_range)
        column_str = "under_bear_acc1_radial"
        df_fft_filt[column_str] = self.lowpass_filter_array(np.array(df_fft[column_str]), self.acc1_max_freq_range)
        column_str = "under_bear_acc1_tang"
        df_fft_filt[column_str] = self.lowpass_filter_array(np.array(df_fft[column_str]), self.acc1_max_freq_range)
        return df_fft_filt

    def get_array_fft(self, input_arr):
        # Calculate FFT
        x_output_fft = fftpack.fft(input_arr)
        freqs = fftpack.fftfreq(len(input_arr)) * self.acc_sample_rate
        return freqs, x_output_fft

    def get_df_fft(self, input_df):
        # Create dataframe 
        df_fft = pd.DataFrame()
        # Filter Accelerometer 1
        column_str = "under_bear_acc1_axial"
        df_fft[column_str + "_y"], df_fft[column_str + "_x"]  = self.get_array_fft(np.array(input_df[column_str]))
        column_str = "under_bear_acc1_radial"
        df_fft[column_str + "_y"], df_fft[column_str + "_x"] = self.get_array_fft(np.array(input_df[column_str]))
        column_str = "under_bear_acc1_tang"
        df_fft[column_str + "_y"], df_fft[column_str + "_x"] = self.get_array_fft(np.array(input_df[column_str]))
        return df_fft
