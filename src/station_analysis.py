import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pandas as pd

def analyze_station_activity(df_feat: pl.DataFrame, station_id: int, show_plots: bool = True):
    """
    Analyze activity patterns for a specific station.
    
    Parameters:
    -----------
    df_feat : pl.DataFrame
        Feature dataset with station data
    station_id : int  
        Station ID to analyze
    show_plots : bool
        Whether to display plots
        
    Returns:
    --------
    dict: Analysis results
    """
    
    # filter data for the specific station
    station_data = df_feat.filter(pl.col("station_id") == station_id)
    
    if station_data.height == 0:
        print(f"❌ Station {station_id} not found in dataset!")
        return None
    
    print(f"🚴‍♂️ Analysis for Station {station_id}")
    print("=" * 50)
    
    # basic statistics
    total_periods = station_data.height
    non_zero_deps = station_data.filter(pl.col("dep_last_DT") > 0).height
    non_zero_arrs = station_data.filter(pl.col("arr_last_DT") > 0).height
    
    print(f"📊 Basic Statistics:")
    print(f"   Total time periods: {total_periods:,}")
    print(f"   Periods with departures: {non_zero_deps:,} ({non_zero_deps/total_periods*100:.1f}%)")
    print(f"   Periods with arrivals: {non_zero_arrs:,} ({non_zero_arrs/total_periods*100:.1f}%)")
    
    # activity statistics
    dep_stats = station_data.select([
        pl.col("dep_last_DT").sum().alias("total_departures"),
        pl.col("dep_last_DT").mean().alias("avg_dep_per_period"),
        pl.col("dep_last_DT").max().alias("max_dep_per_period"),
        pl.col("arr_last_DT").sum().alias("total_arrivals"),
        pl.col("arr_last_DT").mean().alias("avg_arr_per_period"),
        pl.col("arr_last_DT").max().alias("max_arr_per_period")
    ]).row(0)
    
    print(f"\n🚴 Activity Summary:")
    print(f"   Total departures: {dep_stats[0]:,}")
    print(f"   Total arrivals: {dep_stats[3]:,}")
    print(f"   Avg departures/period: {dep_stats[1]:.2f}")
    print(f"   Avg arrivals/period: {dep_stats[4]:.2f}")
    print(f"   Max departures/period: {dep_stats[2]}")
    print(f"   Max arrivals/period: {dep_stats[5]}")
    
    # convert to pandas for easier plotting
    station_pd = station_data.to_pandas()
    station_pd['hour'] = station_pd['ts_start'].dt.hour
    station_pd['dow'] = station_pd['ts_start'].dt.dayofweek  # 0=Monday
    station_pd['month'] = station_pd['ts_start'].dt.month
    station_pd['date'] = station_pd['ts_start'].dt.date
    
    # create analysis results dict
    results = {
        'station_id': station_id,
        'basic_stats': dep_stats,
        'data': station_pd
    }
    
    if show_plots:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Station {station_id} - Activity Analysis', fontsize=16, fontweight='bold')
        
        # 1. hourly pattern - departures
        hourly_deps = station_pd.groupby('hour')['dep_last_DT'].agg(['sum', 'mean', 'count'])
        axes[0,0].bar(hourly_deps.index, hourly_deps['sum'], alpha=0.7, color='skyblue')
        axes[0,0].set_title('Total Departures by Hour', fontweight='bold')
        axes[0,0].set_xlabel('Hour of Day')
        axes[0,0].set_ylabel('Total Departures')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. hourly pattern - arrivals  
        hourly_arrs = station_pd.groupby('hour')['arr_last_DT'].agg(['sum', 'mean', 'count'])
        axes[0,1].bar(hourly_arrs.index, hourly_arrs['sum'], alpha=0.7, color='lightcoral')
        axes[0,1].set_title('Total Arrivals by Hour', fontweight='bold')
        axes[0,1].set_xlabel('Hour of Day')
        axes[0,1].set_ylabel('Total Arrivals')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. day of week pattern
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        dow_deps = station_pd.groupby('dow')['dep_last_DT'].sum()
        dow_arrs = station_pd.groupby('dow')['arr_last_DT'].sum()
        
        x = np.arange(len(dow_names))
        width = 0.35
        axes[0,2].bar(x - width/2, dow_deps.values, width, label='Departures', alpha=0.7, color='skyblue')
        axes[0,2].bar(x + width/2, dow_arrs.values, width, label='Arrivals', alpha=0.7, color='lightcoral')
        axes[0,2].set_title('Activity by Day of Week', fontweight='bold')
        axes[0,2].set_xlabel('Day of Week')
        axes[0,2].set_ylabel('Total Trips')
        axes[0,2].set_xticks(x)
        axes[0,2].set_xticklabels(dow_names)
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. heatmap: hour vs day of week for departures
        pivot_deps = station_pd.pivot_table(values='dep_last_DT', index='dow', columns='hour', aggfunc='mean', fill_value=0)
        sns.heatmap(pivot_deps, ax=axes[1,0], cmap='Blues', cbar_kws={'label': 'Avg Departures'})
        axes[1,0].set_title('Departure Heatmap (Hour vs Day)', fontweight='bold')
        axes[1,0].set_xlabel('Hour of Day')
        axes[1,0].set_ylabel('Day of Week')
        axes[1,0].set_yticklabels(dow_names, rotation=0)
        
        # 5. monthly trend
        monthly = station_pd.groupby('month').agg({
            'dep_last_DT': 'sum',
            'arr_last_DT': 'sum'
        })
        axes[1,1].plot(monthly.index, monthly['dep_last_DT'], 'o-', label='Departures', linewidth=2, markersize=6)
        axes[1,1].plot(monthly.index, monthly['arr_last_DT'], 's-', label='Arrivals', linewidth=2, markersize=6)
        axes[1,1].set_title('Monthly Activity Trend', fontweight='bold')
        axes[1,1].set_xlabel('Month')
        axes[1,1].set_ylabel('Total Trips')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        axes[1,1].set_xticks(range(1, 13))
        
        # 6. activity distribution
        non_zero_deps = station_pd[station_pd['dep_last_DT'] > 0]['dep_last_DT']
        if len(non_zero_deps) > 0:
            axes[1,2].hist(non_zero_deps, bins=min(30, int(non_zero_deps.max())), alpha=0.7, color='skyblue', edgecolor='black')
            axes[1,2].set_title('Distribution of Departures per Period\n(excluding zeros)', fontweight='bold')
            axes[1,2].set_xlabel('Departures per 30-min Period')
            axes[1,2].set_ylabel('Frequency')
            axes[1,2].grid(True, alpha=0.3)
        else:
            axes[1,2].text(0.5, 0.5, 'No departures\nrecorded', ha='center', va='center', transform=axes[1,2].transAxes, fontsize=12)
            axes[1,2].set_title('Distribution of Departures per Period', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # additional insights
        print(f"\n🔍 Key Insights:")
        
        # peak hours
        peak_dep_hour = hourly_deps['sum'].idxmax()
        peak_arr_hour = hourly_arrs['sum'].idxmax()
        print(f"   Peak departure hour: {peak_dep_hour}:00 ({hourly_deps.loc[peak_dep_hour, 'sum']} total departures)")
        print(f"   Peak arrival hour: {peak_arr_hour}:00 ({hourly_arrs.loc[peak_arr_hour, 'sum']} total arrivals)")
        
        # peak days
        peak_dep_dow = dow_deps.idxmax()
        peak_arr_dow = dow_arrs.idxmax()
        print(f"   Peak departure day: {dow_names[peak_dep_dow]} ({dow_deps.iloc[peak_dep_dow]} total departures)")
        print(f"   Peak arrival day: {dow_names[peak_arr_dow]} ({dow_arrs.iloc[peak_arr_dow]} total arrivals)")
        
        # balance analysis
        total_deps = station_pd['dep_last_DT'].sum()
        total_arrs = station_pd['arr_last_DT'].sum()
        balance = total_arrs - total_deps
        print(f"   Station balance: {balance:+,} (positive = net gain, negative = net loss)")
        
        # gender distribution (for periods with activity)
        active_periods = station_pd[station_pd['dep_last_DT'] > 0]
        if len(active_periods) > 0:
            avg_share_male = active_periods['share_male'].mean()
            avg_share_female = active_periods['share_female'].mean()
            avg_share_other = active_periods['share_other'].mean()
            print(f"   Gender distribution (when active): Male {avg_share_male:.1%}, Female {avg_share_female:.1%}, Other {avg_share_other:.1%}")
        
        # activity concentration
        active_hours = (hourly_deps['sum'] > 0).sum()
        print(f"   Active hours per day: {active_hours}/24 ({active_hours/24*100:.1f}%)")
        
        results.update({
            'hourly_departures': hourly_deps,
            'hourly_arrivals': hourly_arrs,
            'dow_pattern': {'departures': dow_deps, 'arrivals': dow_arrs},
            'monthly_trend': monthly,
            'peak_hours': {'dep': peak_dep_hour, 'arr': peak_arr_hour},
            'peak_days': {'dep': peak_dep_dow, 'arr': peak_arr_dow},
            'balance': balance
        })
    
    return results

# función auxiliar para comparar múltiples estaciones
def compare_stations(df_feat: pl.DataFrame, station_ids: list, metric: str = 'dep_last_DT'):
    """
    Compare activity patterns across multiple stations.
    
    Parameters:
    -----------
    df_feat : pl.DataFrame
        Feature dataset
    station_ids : list
        List of station IDs to compare  
    metric : str
        Metric to compare ('dep_last_DT', 'arr_last_DT', etc.)
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Station Comparison - {metric}', fontsize=16, fontweight='bold')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(station_ids)))
    
    for i, station_id in enumerate(station_ids):
        station_data = df_feat.filter(pl.col("station_id") == station_id).to_pandas()
        if len(station_data) == 0:
            continue
            
        station_data['hour'] = station_data['ts_start'].dt.hour
        station_data['dow'] = station_data['ts_start'].dt.dayofweek
        
        # hourly pattern
        hourly = station_data.groupby('hour')[metric].sum()
        axes[0,0].plot(hourly.index, hourly.values, 'o-', label=f'Station {station_id}', 
                      color=colors[i], linewidth=2, markersize=4)
        
        # daily pattern
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        daily = station_data.groupby('dow')[metric].sum()
        axes[0,1].plot(range(7), daily.values, 's-', label=f'Station {station_id}',
                      color=colors[i], linewidth=2, markersize=6)
        
        # monthly pattern
        monthly = station_data.groupby(station_data['ts_start'].dt.month)[metric].sum()
        axes[1,0].plot(monthly.index, monthly.values, '^-', label=f'Station {station_id}',
                      color=colors[i], linewidth=2, markersize=5)
        
        # activity distribution (non-zero values)
        non_zero = station_data[station_data[metric] > 0][metric]
        if len(non_zero) > 0:
            axes[1,1].hist(non_zero, bins=20, alpha=0.6, label=f'Station {station_id}',
                          color=colors[i], density=True)
    
    # configure subplots
    axes[0,0].set_title('Hourly Pattern')
    axes[0,0].set_xlabel('Hour of Day')
    axes[0,0].set_ylabel(f'Total {metric}')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].set_title('Daily Pattern')
    axes[0,1].set_xlabel('Day of Week')
    axes[0,1].set_ylabel(f'Total {metric}')
    axes[0,1].set_xticks(range(7))
    axes[0,1].set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].set_title('Monthly Pattern')
    axes[1,0].set_xlabel('Month')
    axes[1,0].set_ylabel(f'Total {metric}')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].set_title('Activity Distribution')
    axes[1,1].set_xlabel(f'{metric} per Period')
    axes[1,1].set_ylabel('Density')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

