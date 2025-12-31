import numpy as np
from windrose import WindroseAxes
import matplotlib.pyplot as plt
from netCDF4 import Dataset, num2date
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib.ticker as ticker

regiao = 'se'

rootgrp = Dataset(f'/home/numa23/Public/Projeto_BR_OTEC/copernicus/dados_wrf/dados_finais_com_landmask/serie_temporal_ventomax_{regiao}.nc','r')

vars = rootgrp.variables

wspd = vars['wspdfinal'][:]
solarf = vars['solarfinal'][:]
wdir = vars['wdirfinal'][:]

lat = vars['latitude'][:]
lon = vars['longitude'][:]
time = vars['timerun'][:]

# Get the time information for the file.
timeUnits = vars['timerun'].units

# Make dates for the file - convert cftime to regular datetime
Date = num2date(time, timeUnits, calendar='standard')

# Convert cftime objects to regular datetime objects
datetime_array = np.array([datetime(d.year, d.month, d.day, d.hour, d.minute, d.second) 
                          for d in Date])

Month = np.asarray([d.month for d in Date])
Year = np.asarray([d.year for d in Date])

figsdir = "/home/numa23/Public/Projeto_BR_OTEC/copernicus/figures/"

# Create a figure with larger width to accommodate bigger time series
fig = plt.figure(figsize=(18, 8))

# ------------------------------------------------------------
# Subplot 1: Wind Rose (smaller, left side)
# ------------------------------------------------------------
# Smaller wind rose: reduced width from 0.4 to 0.3, increased left margin
ax1 = WindroseAxes(fig, rect=[0.05, 0.1, 0.28, 0.8])
fig.add_axes(ax1)

# Define custom wind speed bins: 0-2.5, 2.5-5, 5-7.5, 7.5-10, >10
bins = [0, 2.5, 5, 7.5, 10]  # Add small buffer for max value

# Plot the wind rose with custom bins
ax1.bar(wdir, wspd, normed=True, opening=0.8, edgecolor='white', bins=bins)

# Force the radial axis to show up to 55%
ax1.set_ylim(0, 55)

# Set radial ticks and labels with percentage signs
radial_ticks = [5, 15, 25, 35, 45, 55]
ax1.set_yticks(radial_ticks)
ax1.set_yticklabels(['5%', '15%', '25%', '35%', '45%', '55%'])

# INCREASE FONT SIZE FOR PERCENTAGE LABELS (radial labels)
for label in ax1.get_yticklabels():
    label.set_fontsize(10)  # Increase from default
    label.set_fontweight('normal')
    

# INCREASE FONT SIZE FOR DIRECTIONAL LABELS (N, NE, E, etc.)
# Get the current directional labels
for label in ax1.get_xticklabels():
    label.set_fontsize(13)  # Increase from default
    label.set_fontweight('normal')


# Move legend further left to avoid overlap and make it larger
# Create custom legend labels based on our bins

ax1.set_legend(title=r"Wind Speed (m s$\rm^{-1}$)", 
               bbox_to_anchor=(-0.15, -0.35),  # Adjusted position
               loc='lower left',
               prop={'size': 15},  # Increase font size
               title_fontsize=15)  # Increase title font size
               

# NOW modify the legend after it's created
legend = ax1.get_legend()
if legend:    
    
    # Set label fontsize
    for text in legend.get_texts():
        text.set_fontsize(15)

ax1.set_title(f"Wind Rose - {regiao.upper()} Region", fontsize=14, pad=30)

# ------------------------------------------------------------
# Subplot 2: Time Series (larger, right side)
# ------------------------------------------------------------
# Larger time series: increased width from 0.4 to 0.6, moved left
ax2 = fig.add_axes([0.38, 0.1, 0.5, 0.8])

# Plot wind speed time series with slightly thicker line
ax2.plot(datetime_array, wspd, 'b-', linewidth=1.2, alpha=0.8, label='Wind Speed')

# Fix date formatting - use AutoDateLocator for better label spacing
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Calculate appropriate interval based on data span
if len(datetime_array) > 1:
    # Calculate total time span in months
    start_date = datetime_array[0]
    end_date = datetime_array[-1]
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    
    # Determine optimal interval - show about 8-12 labels
    if total_months <= 12:
        interval = 1  # Monthly for 1 year or less
    elif total_months <= 24:
        interval = 2  # Every 2 months for 2 years
    elif total_months <= 48:
        interval = 4  # Every 4 months for 4 years
    elif total_months <= 96:
        interval = 6  # Every 6 months for 8 years
    else:
        interval = 12  # Yearly for longer periods
    
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=interval))
    
    # Ensure we have enough ticks - add more if needed
    if len(ax2.get_xticks()) < 3:
        # If we still have too few ticks, force more
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))
else:
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=15))

# Rotate x-axis labels for better readability
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=15)

# Add minor grid for x-axis (months)
ax2.xaxis.set_minor_locator(mdates.MonthLocator())

# INCREASE FONT SIZE FOR Y-AXIS TICK LABELS (NUMBERS)
ax2.tick_params(axis='y', labelsize=13)

# Add labels and grid with larger font
ax2.set_ylabel(r'Wind Speed (m s$\rm^{-1}$)', fontsize=16, color='k')
ax2.set_title(f'Wind Speed Time Series - {regiao.upper()} Region', fontsize=15, pad=15)
ax2.grid(True, alpha=0.3, which='major')
ax2.grid(True, alpha=0.2, which='minor')


# Set y-axis limits with some padding
wspd_min, wspd_max = wspd.min(), wspd.max()
ax2.set_ylim(max(0, wspd_min * 0.95), wspd_max * 1.05)

# Add y-axis minor grid
ax2.yaxis.set_minor_locator(ticker.AutoMinorLocator())
ax2.grid(True, which='minor', axis='y', alpha=0.2)

# Adjust layout to prevent label overlap
plt.tight_layout(rect=[0, 0, 0.95, 1])  # Adjust right margin for better spacing

# Save the figure
nameoffigure = f"windrose_timeseries_{regiao}_wrf"
string_in_string = "{}".format(nameoffigure)
plt.savefig(figsdir + string_in_string, dpi=300, bbox_inches='tight')
plt.show()
plt.close(fig)

print(f"Figure saved as: {figsdir}{string_in_string}.png")
print(f"Date range: {datetime_array[0]} to {datetime_array[-1]}")
print(f"Total months: {(datetime_array[-1].year - datetime_array[0].year) * 12 + (datetime_array[-1].month - datetime_array[0].month)}")
print(f"Wind Speed Statistics:")
print(f"  Mean: {np.mean(wspd):.2f} m/s")
print(f"  Std Dev: {np.std(wspd):.2f} m/s")
print(f"  Max: {np.max(wspd):.2f} m/s")
print(f"  Min: {np.min(wspd):.2f} m/s")
