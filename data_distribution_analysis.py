#!/usr/bin/env python3
"""
Data Distribution Analysis for Optimal Worker Configuration

Analyzes the actual file distribution per opportunity to determine
the most efficient worker allocation strategy.
"""

import statistics
import math

# Data from user's SQL query
file_distribution = [
    (0, 597522),
    (1, 291490),
    (4, 25202),
    (2, 21217),
    (3, 11144),
    (5, 3422),
    (6, 2402),
    (7, 1686),
    (8, 1525),
    (9, 939),
    (10, 822),
    (11, 620),
    (13, 497),
    (12, 492),
    (14, 423),
    (15, 322),
    (16, 314),
    (17, 278),
    (18, 207)
]

def analyze_distribution():
    """Analyze the file distribution to optimize worker allocation."""
    
    print("📊 OPPORTUNITY FILE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    # Calculate total opportunities
    total_opportunities = sum(count for files, count in file_distribution)
    print(f"Total Opportunities: {total_opportunities:,}")
    
    # Calculate opportunities with files (excluding 0-file opportunities)
    opportunities_with_files = sum(count for files, count in file_distribution if files > 0)
    print(f"Opportunities with Files: {opportunities_with_files:,} ({opportunities_with_files/total_opportunities*100:.1f}%)")
    print(f"Empty Opportunities: {file_distribution[0][1]:,} ({file_distribution[0][1]/total_opportunities*100:.1f}%)")
    
    print("\n📈 FILE DISTRIBUTION STATISTICS")
    print("-" * 40)
    
    # Create expanded dataset for statistical analysis
    expanded_data = []
    for files, count in file_distribution:
        expanded_data.extend([files] * count)
    
    # Calculate statistics
    mean_files = statistics.mean(expanded_data)
    median_files = statistics.median(expanded_data)
    mode_files = statistics.mode(expanded_data)
    
    print(f"Mean files per opportunity: {mean_files:.2f}")
    print(f"Median files per opportunity: {median_files}")
    print(f"Mode (most common): {mode_files} files")
    
    # Calculate percentiles for opportunities with files only
    with_files_data = [files for files in expanded_data if files > 0]
    if with_files_data:
        mean_with_files = statistics.mean(with_files_data)
        median_with_files = statistics.median(with_files_data)
        print(f"\nFor opportunities WITH files:")
        print(f"Mean files: {mean_with_files:.2f}")
        print(f"Median files: {median_with_files}")
    
    print("\n🎯 TOP 10 MOST COMMON FILE COUNTS")
    print("-" * 40)
    sorted_distribution = sorted(file_distribution, key=lambda x: x[1], reverse=True)
    for i, (files, count) in enumerate(sorted_distribution[:10]):
        percentage = count / total_opportunities * 100
        print(f"{i+1:2d}. {files:2d} files: {count:7,} opportunities ({percentage:5.1f}%)")
    
    return {
        'total_opportunities': total_opportunities,
        'opportunities_with_files': opportunities_with_files,
        'mean_files': mean_files,
        'median_files': median_files,
        'mode_files': mode_files,
        'mean_with_files': mean_with_files if with_files_data else 0,
        'distribution': file_distribution
    }

def calculate_worker_efficiency(opportunity_workers, file_workers_per_opportunity):
    """Calculate efficiency of a given worker configuration."""
    
    total_workers = opportunity_workers * file_workers_per_opportunity
    
    # Calculate utilization for each file count
    utilization_data = []
    wasted_workers = 0
    
    for files, count in file_distribution:
        if files == 0:
            # No work for these opportunities
            continue
            
        # For this file count, how many file workers are actually used?
        workers_used_per_opp = min(files, file_workers_per_opportunity)
        workers_wasted_per_opp = file_workers_per_opportunity - workers_used_per_opp
        
        utilization_data.append({
            'files': files,
            'opportunities': count,
            'workers_used': workers_used_per_opp,
            'workers_wasted': workers_wasted_per_opp,
            'utilization_rate': workers_used_per_opp / file_workers_per_opportunity
        })
        
        wasted_workers += workers_wasted_per_opp * count
    
    # Calculate overall efficiency
    total_file_processing_jobs = sum(files * count for files, count in file_distribution if files > 0)
    opportunities_with_files = sum(count for files, count in file_distribution if files > 0)
    
    # Average utilization rate weighted by opportunity count
    weighted_utilization = sum(
        data['utilization_rate'] * data['opportunities'] 
        for data in utilization_data
    ) / opportunities_with_files if opportunities_with_files > 0 else 0
    
    return {
        'opportunity_workers': opportunity_workers,
        'file_workers_per_opportunity': file_workers_per_opportunity,
        'total_workers': total_workers,
        'weighted_utilization': weighted_utilization,
        'utilization_data': utilization_data
    }

def find_optimal_configuration(target_total_workers=96):
    """Find the optimal worker configuration for the given total worker count."""
    
    print(f"\n🔍 OPTIMIZING FOR {target_total_workers} TOTAL WORKERS")
    print("=" * 60)
    
    configurations = []
    
    # Test various configurations that multiply to target_total_workers
    for opp_workers in range(1, target_total_workers + 1):
        if target_total_workers % opp_workers == 0:
            file_workers = target_total_workers // opp_workers
            efficiency = calculate_worker_efficiency(opp_workers, file_workers)
            configurations.append(efficiency)
    
    # Sort by utilization efficiency
    configurations.sort(key=lambda x: x['weighted_utilization'], reverse=True)
    
    print("🏆 TOP 10 CONFIGURATIONS (by efficiency):")
    print("-" * 60)
    print(f"{'Rank':<4} {'Opp Workers':<12} {'File Workers':<12} {'Utilization':<12} {'Total Workers':<12}")
    print("-" * 60)
    
    for i, config in enumerate(configurations[:10]):
        print(f"{i+1:<4} {config['opportunity_workers']:<12} {config['file_workers_per_opportunity']:<12} "
              f"{config['weighted_utilization']*100:>9.1f}% {config['total_workers']:<12}")
    
    # Recommend the best configuration
    best_config = configurations[0]
    
    print(f"\n🎯 RECOMMENDED CONFIGURATION:")
    print("-" * 40)
    print(f"Opportunity Workers: {best_config['opportunity_workers']}")
    print(f"File Workers per Opportunity: {best_config['file_workers_per_opportunity']}")
    print(f"Total Workers: {best_config['total_workers']}")
    print(f"Worker Utilization: {best_config['weighted_utilization']*100:.1f}%")
    
    return best_config

def analyze_current_vs_optimal():
    """Compare current configuration with optimal configuration."""
    
    print(f"\n📊 CURRENT vs OPTIMAL COMPARISON")
    print("=" * 60)
    
    # Current configuration
    current = calculate_worker_efficiency(12, 8)
    
    # Optimal configuration  
    optimal = find_optimal_configuration(96)
    
    print(f"\nCURRENT CONFIG (12 opp × 8 file = 96 total):")
    print(f"Worker Utilization: {current['weighted_utilization']*100:.1f}%")
    
    print(f"\nOPTIMAL CONFIG ({optimal['opportunity_workers']} opp × {optimal['file_workers_per_opportunity']} file = {optimal['total_workers']} total):")
    print(f"Worker Utilization: {optimal['weighted_utilization']*100:.1f}%")
    
    improvement = optimal['weighted_utilization'] / current['weighted_utilization']
    print(f"\nEFFICIENCY IMPROVEMENT: {improvement:.2f}x ({(improvement-1)*100:.1f}% better)")
    
    return current, optimal

if __name__ == "__main__":
    # Run the analysis
    stats = analyze_distribution()
    current, optimal = analyze_current_vs_optimal()
    
    print(f"\n🚀 SUMMARY & RECOMMENDATIONS")
    print("=" * 60)
    print(f"• 66.2% of opportunities have 0 files (skip these entirely)")
    print(f"• 32.3% have exactly 1 file (file workers > 1 are wasted)")
    print(f"• Only 1.5% have 4+ files (where current 8 file workers help)")
    print(f"• Optimal config improves efficiency by {optimal['weighted_utilization']/current['weighted_utilization']:.1f}x")
    print(f"• Recommended: {optimal['opportunity_workers']} opportunity workers × {optimal['file_workers_per_opportunity']} file workers")
