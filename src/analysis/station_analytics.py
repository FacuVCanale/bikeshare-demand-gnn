"""
Station analytics and visualization functions for EcoBici data.

This module provides functionality to:
- Analyze activity patterns for individual stations
- Compare multiple stations
- Generate visualizations for station usage
- Extract insights about temporal patterns
"""

import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import pandas as pd
from typing import Dict, List, Any, Optional


def analyze_station_activity(df_feat: pl.DataFrame, station_id: int, show_plots: bool = True) -> Optional[Dict[str, Any]]:
    """
    Analyze activity patterns for a specific station.
    
    Args:
        df_feat: Feature dataset with station data
        station_id: Station ID to analyze
        show_plots: Whether to display plots
        
    Returns:
        Dictionary with analysis results or None if station not found
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
        if len(active_periods) > 0 and 'share_male' in active_periods.columns:
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


def compare_stations(df_feat: pl.DataFrame, station_ids: List[int], metric: str = 'dep_last_DT') -> Dict[str, Any]:
    """
    Compare activity patterns across multiple stations.
    
    Args:
        df_feat: Feature dataset with station data
        station_ids: List of station IDs to compare
        metric: Metric to compare ('dep_last_DT' or 'arr_last_DT')
        
    Returns:
        Dictionary with comparison results
    """
    print(f"🔄 Comparing {len(station_ids)} stations on metric: {metric}")
    print("=" * 50)
    
    comparison_data = []
    
    for station_id in station_ids:
        station_data = df_feat.filter(pl.col("station_id") == station_id)
        
        if station_data.height == 0:
            print(f"❌ Station {station_id} not found - skipping")
            continue
            
        # calculate key metrics
        stats = station_data.select([
            pl.lit(station_id).alias("station_id"),
            pl.col(metric).sum().alias("total"),
            pl.col(metric).mean().alias("mean_per_period"),
            pl.col(metric).max().alias("max_per_period"),
            (pl.col(metric) > 0).sum().alias("active_periods"),
            pl.col(metric).std().alias("std_per_period")
        ]).row(0)
        
        comparison_data.append({
            'station_id': stats[0],
            'total': stats[1], 
            'mean_per_period': stats[2],
            'max_per_period': stats[3],
            'active_periods': stats[4],
            'std_per_period': stats[5],
            'activity_rate': stats[4] / station_data.height * 100
        })
    
    # convert to DataFrame for easier analysis
    comp_df = pd.DataFrame(comparison_data)
    
    if len(comp_df) == 0:
        print("❌ No valid stations found for comparison")
        return {}
    
    # sort by total activity
    comp_df = comp_df.sort_values('total', ascending=False)
    
    # print comparison
    print(f"📊 Station Comparison Results ({metric}):")
    print(f"{'Station':<10} {'Total':<8} {'Mean/Period':<12} {'Max/Period':<11} {'Activity Rate':<12}")
    print("-" * 60)
    
    for _, row in comp_df.iterrows():
        print(f"{row['station_id']:<10} {row['total']:<8,.0f} {row['mean_per_period']:<12.2f} {row['max_per_period']:<11.0f} {row['activity_rate']:<12.1f}%")
    
    # create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Station Comparison - {metric}', fontsize=16, fontweight='bold')
    
    # 1. total activity bar chart
    axes[0,0].bar(range(len(comp_df)), comp_df['total'], alpha=0.7, color='skyblue')
    axes[0,0].set_title('Total Activity by Station')
    axes[0,0].set_xlabel('Station Rank')
    axes[0,0].set_ylabel(f'Total {metric}')
    axes[0,0].set_xticks(range(len(comp_df)))
    axes[0,0].set_xticklabels([f"#{i+1}\n{int(sid)}" for i, sid in enumerate(comp_df['station_id'])], rotation=45)
    
    # 2. mean activity per period
    axes[0,1].bar(range(len(comp_df)), comp_df['mean_per_period'], alpha=0.7, color='lightcoral')
    axes[0,1].set_title('Mean Activity per Period')
    axes[0,1].set_xlabel('Station Rank')
    axes[0,1].set_ylabel(f'Mean {metric} per Period')
    axes[0,1].set_xticks(range(len(comp_df)))
    axes[0,1].set_xticklabels([f"#{i+1}\n{int(sid)}" for i, sid in enumerate(comp_df['station_id'])], rotation=45)
    
    # 3. activity rate (percentage of periods with activity)
    axes[1,0].bar(range(len(comp_df)), comp_df['activity_rate'], alpha=0.7, color='lightgreen')
    axes[1,0].set_title('Activity Rate (%)')
    axes[1,0].set_xlabel('Station Rank')
    axes[1,0].set_ylabel('% Periods with Activity')
    axes[1,0].set_xticks(range(len(comp_df)))
    axes[1,0].set_xticklabels([f"#{i+1}\n{int(sid)}" for i, sid in enumerate(comp_df['station_id'])], rotation=45)
    
    # 4. scatter plot: mean vs max activity
    axes[1,1].scatter(comp_df['mean_per_period'], comp_df['max_per_period'], alpha=0.7, s=60)
    axes[1,1].set_title('Mean vs Max Activity')
    axes[1,1].set_xlabel('Mean per Period')
    axes[1,1].set_ylabel('Max per Period')
    
    # annotate points with station IDs
    for _, row in comp_df.iterrows():
        axes[1,1].annotate(str(int(row['station_id'])), 
                          (row['mean_per_period'], row['max_per_period']),
                          xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # summary insights
    print(f"\n🔍 Comparison Insights:")
    top_station = comp_df.iloc[0]
    print(f"   Most active: Station {int(top_station['station_id'])} ({top_station['total']:,.0f} total)")
    print(f"   Highest mean: Station {int(comp_df.loc[comp_df['mean_per_period'].idxmax(), 'station_id'])} ({comp_df['mean_per_period'].max():.2f} per period)")
    print(f"   Most consistent: Station {int(comp_df.loc[comp_df['activity_rate'].idxmax(), 'station_id'])} ({comp_df['activity_rate'].max():.1f}% activity rate)")
    
    return {
        'comparison_data': comp_df,
        'metric': metric,
        'summary': {
            'most_active': int(top_station['station_id']),
            'highest_mean': int(comp_df.loc[comp_df['mean_per_period'].idxmax(), 'station_id']),
            'most_consistent': int(comp_df.loc[comp_df['activity_rate'].idxmax(), 'station_id'])
        }
    }


def analyze_temporal_patterns(df_feat: pl.DataFrame, station_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Analyze temporal patterns across stations or for specific stations.
    
    Args:
        df_feat: Feature dataset with station data
        station_ids: Optional list of specific stations to analyze. If None, analyzes all stations
        
    Returns:
        Dictionary with temporal pattern analysis
    """
    print("🕐 Analyzing temporal patterns...")
    
    # filter data if specific stations requested
    if station_ids is not None:
        data = df_feat.filter(pl.col("station_id").is_in(station_ids))
        print(f"   Analyzing {len(station_ids)} specific stations")
    else:
        data = df_feat
        print(f"   Analyzing all {data['station_id'].n_unique()} stations")
    
    # convert to pandas for temporal analysis
    df_pd = data.to_pandas()
    df_pd['hour'] = df_pd['ts_start'].dt.hour
    df_pd['dow'] = df_pd['ts_start'].dt.dayofweek  # 0=Monday
    df_pd['month'] = df_pd['ts_start'].dt.month
    
    # aggregate by temporal dimensions
    hourly_pattern = df_pd.groupby('hour').agg({
        'dep_last_DT': ['sum', 'mean'],
        'arr_last_DT': ['sum', 'mean']
    }).round(2)
    
    dow_pattern = df_pd.groupby('dow').agg({
        'dep_last_DT': ['sum', 'mean'],
        'arr_last_DT': ['sum', 'mean']
    }).round(2)
    
    monthly_pattern = df_pd.groupby('month').agg({
        'dep_last_DT': ['sum', 'mean'],
        'arr_last_DT': ['sum', 'mean']
    }).round(2)
    
    # find peak patterns
    peak_dep_hour = hourly_pattern[('dep_last_DT', 'sum')].idxmax()
    peak_arr_hour = hourly_pattern[('arr_last_DT', 'sum')].idxmax()
    peak_dep_dow = dow_pattern[('dep_last_DT', 'sum')].idxmax()
    peak_arr_dow = dow_pattern[('arr_last_DT', 'sum')].idxmax()
    
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    print(f"\n📈 Temporal Pattern Summary:")
    print(f"   Peak departure hour: {peak_dep_hour}:00")
    print(f"   Peak arrival hour: {peak_arr_hour}:00") 
    print(f"   Peak departure day: {dow_names[peak_dep_dow]}")
    print(f"   Peak arrival day: {dow_names[peak_arr_dow]}")
    
    return {
        'hourly_patterns': hourly_pattern,
        'dow_patterns': dow_pattern,
        'monthly_patterns': monthly_pattern,
        'peaks': {
            'departure_hour': peak_dep_hour,
            'arrival_hour': peak_arr_hour,
            'departure_dow': peak_dep_dow,
            'arrival_dow': peak_arr_dow
        }
    }


def station_ranking_analysis(df_feat: pl.DataFrame, top_n: int = 20) -> Dict[str, Any]:
    """
    Create comprehensive ranking analysis of stations.
    
    Args:
        df_feat: Feature dataset with station data
        top_n: Number of top stations to include in analysis
        
    Returns:
        Dictionary with ranking analysis results
    """
    print(f"🏆 Station Ranking Analysis (Top {top_n})")
    print("=" * 50)
    
    # calculate comprehensive metrics for each station
    station_metrics = (
        df_feat.group_by("station_id")
        .agg([
            pl.col("dep_last_DT").sum().alias("total_departures"),
            pl.col("arr_last_DT").sum().alias("total_arrivals"),
            pl.col("dep_last_DT").mean().alias("avg_dep_per_period"),
            pl.col("arr_last_DT").mean().alias("avg_arr_per_period"),
            pl.col("dep_last_DT").max().alias("max_dep_per_period"),
            pl.col("arr_last_DT").max().alias("max_arr_per_period"),
            (pl.col("dep_last_DT") > 0).sum().alias("active_dep_periods"),
            (pl.col("arr_last_DT") > 0).sum().alias("active_arr_periods"),
            pl.col("dep_last_DT").std().alias("dep_variability"),
            pl.col("arr_last_DT").std().alias("arr_variability"),
            pl.len().alias("total_periods")
        ])
        .with_columns([
            (pl.col("total_arrivals") - pl.col("total_departures")).alias("net_balance"),
            (pl.col("active_dep_periods") / pl.col("total_periods") * 100).alias("dep_activity_rate"),
            (pl.col("active_arr_periods") / pl.col("total_periods") * 100).alias("arr_activity_rate"),
            (pl.col("total_departures") + pl.col("total_arrivals")).alias("total_activity")
        ])
        .sort("total_activity", descending=True)
        .head(top_n)
    )
    
    # convert to pandas for easier analysis
    metrics_df = station_metrics.to_pandas()
    
    print(f"📊 Top {len(metrics_df)} Stations by Total Activity:")
    print(f"{'Rank':<5} {'Station':<8} {'Total Act.':<10} {'Departures':<11} {'Arrivals':<9} {'Balance':<8} {'Dep Rate':<9}")
    print("-" * 70)
    
    for i, row in metrics_df.iterrows():
        print(f"{i+1:<5} {int(row['station_id']):<8} {row['total_activity']:<10,.0f} {row['total_departures']:<11,.0f} {row['total_arrivals']:<9,.0f} {row['net_balance']:<8,.0f} {row['dep_activity_rate']:<9.1f}%")
    
    # categorize stations
    high_activity = metrics_df[metrics_df['total_activity'] > metrics_df['total_activity'].quantile(0.8)]
    balanced_stations = metrics_df[abs(metrics_df['net_balance']) < metrics_df['total_activity'] * 0.1]
    departure_hubs = metrics_df[metrics_df['net_balance'] < -metrics_df['total_activity'] * 0.1]
    arrival_hubs = metrics_df[metrics_df['net_balance'] > metrics_df['total_activity'] * 0.1]
    
    print(f"\n🏷️  Station Categories:")
    print(f"   High Activity Stations: {len(high_activity)}")
    print(f"   Balanced Stations: {len(balanced_stations)}")
    print(f"   Departure Hubs: {len(departure_hubs)}")
    print(f"   Arrival Hubs: {len(arrival_hubs)}")
    
    return {
        'rankings': metrics_df,
        'categories': {
            'high_activity': high_activity['station_id'].tolist(),
            'balanced': balanced_stations['station_id'].tolist(),
            'departure_hubs': departure_hubs['station_id'].tolist(),
            'arrival_hubs': arrival_hubs['station_id'].tolist()
        },
        'summary_stats': {
            'total_stations_analyzed': len(metrics_df),
            'total_activity_sum': metrics_df['total_activity'].sum(),
            'avg_activity_per_station': metrics_df['total_activity'].mean(),
            'most_active_station': int(metrics_df.iloc[0]['station_id'])
        }
    } 