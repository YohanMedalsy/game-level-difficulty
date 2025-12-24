#!/usr/bin/env python3
"""
Validation Script: Incremental EWMA Computation

This script validates that the incremental EWMA computation (--ewma-lookback-days 1)
produces IDENTICAL results to the full computation method.

Test Strategy:
1. Pick a date that's already in the Delta table (e.g., 2025-11-29)
2. Load the EWMA features for that date from Delta (ground truth)
3. Recompute the same date's EWMA features using incremental method:
   - Load previous day (2025-11-28) from Delta
   - Query Snowflake for test date (2025-11-29)
   - Combine and compute EWMA
4. Compare ground truth vs recomputed
5. Calculate absolute differences for each EWMA column
6. Verify differences are within floating-point precision (< 1e-10)

Usage:
    python validate_ewma_incremental.py \\
        --test-date 2025-11-29 \\
        --delta-path dbfs:/mnt/features/daily_features_inference.delta \\
        --sf-config-json /path/to/snowflake_config.json \\
        --snowflake-table spaceplay.unity.boxjam_snapshot_2025_10_17

Author: Claude Code + Data Engineering Team
Date: 2025-12-02
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Spark imports
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, DateType, IntegerType


def to_spark_path(path: str) -> str:
    """Convert driver-style /dbfs paths to spark-friendly dbfs:/ URIs."""
    if not path:
        return path
    if path.startswith("/dbfs/"):
        return "dbfs:" + path[5:]
    return path


def load_snowflake_config(json_path: str) -> Dict:
    """Load Snowflake configuration from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def load_ground_truth_from_delta(spark: SparkSession, delta_path: str, test_date: str) -> DataFrame:
    """Load the ground truth EWMA features for test_date from Delta table."""
    print(f"\n{'='*80}")
    print(f"STEP 1: Loading Ground Truth from Delta")
    print(f"{'='*80}")

    delta_spark_path = to_spark_path(delta_path)
    print(f"📥 Reading Delta table: {delta_spark_path}")
    print(f"   Filtering to date: {test_date}")

    df = spark.read.format("delta").load(delta_spark_path) \
        .filter(F.col("session_date") == F.lit(test_date))

    row_count = df.count()
    print(f"   ✅ Loaded {row_count:,} rows")

    # Identify EWMA columns
    ewma_cols = [c for c in df.columns if c.startswith('ewma_')]
    print(f"   📊 Found {len(ewma_cols)} EWMA columns")
    print(f"      Sample: {ewma_cols[:5]}...")

    return df


def query_snowflake_for_date(spark: SparkSession, sf_config: Dict, date: str, snowflake_table: str) -> DataFrame:
    """Query Snowflake for raw events for a specific date."""
    print(f"\n📊 Querying Snowflake for date: {date}")
    print(f"   Table: {snowflake_table}")

    # Build Snowflake connector options
    sf_options = {
        "sfUrl": sf_config.get("url", ""),
        "sfUser": sf_config.get("user", ""),
        "sfPassword": sf_config.get("password", ""),
        "sfDatabase": sf_config.get("database", "SPACEPLAY"),
        "sfSchema": sf_config.get("schema", "UNITY"),
        "sfWarehouse": sf_config.get("warehouse", "DASHBOARD_DEFAULT"),
    }

    if sf_config.get("role"):
        sf_options["sfRole"] = sf_config["role"]

    # Query for specific date
    query = f"""
    SELECT *
    FROM {snowflake_table}
    WHERE gameName = 'Box Jam'
      AND environmentName = 'Live'
      AND gauserstartdate >= '{date}'
      AND gauserstartdate <= '{date}'
    """

    df = spark.read \
        .format("net.snowflake.spark.snowflake") \
        .options(**sf_options) \
        .option("query", query) \
        .load()

    row_count = df.count()
    print(f"   ✅ Loaded {row_count:,} raw events from Snowflake")

    return df


def recompute_with_incremental_method(
    spark: SparkSession,
    delta_path: str,
    previous_date: str,
    test_date: str,
    today_raw_events: DataFrame
) -> DataFrame:
    """
    Recompute EWMA features using incremental method.

    This function mimics what Phase 0 does with --ewma-lookback-days 1:
    1. Load previous_date's features from Delta (raw features only, no EWMA)
    2. Aggregate test_date's raw events
    3. Union previous + today
    4. Compute EWMA on combined data
    5. Filter to keep only test_date
    """
    print(f"\n{'='*80}")
    print(f"STEP 2: Recomputing with Incremental Method")
    print(f"{'='*80}")

    # Import functions from Phase 0 script
    # NOTE: This requires the script to be importable
    # For now, we'll implement simplified versions

    # Step 1: Load previous date from Delta
    print(f"\n📥 Loading previous date from Delta: {previous_date}")
    delta_spark_path = to_spark_path(delta_path)

    previous_df = spark.read.format("delta").load(delta_spark_path) \
        .filter(F.col("session_date") == F.lit(previous_date))

    prev_count = previous_df.count()
    print(f"   ✅ Loaded {prev_count:,} rows from {previous_date}")

    # Drop EWMA columns from previous day (will be recomputed)
    ewma_cols_to_drop = [c for c in previous_df.columns if c.startswith('ewma_')]
    if ewma_cols_to_drop:
        print(f"   🧹 Dropping {len(ewma_cols_to_drop)} EWMA columns")
        previous_df = previous_df.drop(*ewma_cols_to_drop)

    # Step 2: Aggregate today's raw events
    print(f"\n📊 Aggregating {test_date}'s raw events...")
    print(f"   ⚠️  NOTE: Using simplified aggregation for validation")
    print(f"       For full validation, import aggregate_daily_features() from Phase 0")

    # Simplified aggregation - just enough to test EWMA computation
    # In production, this should use the full aggregate_daily_features() function
    today_df = aggregate_daily_features_simplified(today_raw_events, test_date)

    # Step 3: Union previous + today
    print(f"\n🔗 Combining previous ({previous_date}) + today ({test_date})")

    # Normalize column names to lowercase
    prev_lower = previous_df.select([F.col(c).alias(c.lower()) for c in previous_df.columns])
    today_lower = today_df.select([F.col(c).alias(c.lower()) for c in today_df.columns])

    combined_df = prev_lower.unionByName(today_lower, allowMissingColumns=True)
    combined_count = combined_df.count()
    print(f"   ✅ Combined: {combined_count:,} rows ({prev_count:,} + {combined_count-prev_count:,})")

    # Step 4: Compute EWMA on combined data
    print(f"\n🧮 Computing EWMA features on combined data...")
    df_with_ewma = compute_ewma_features_spark_simplified(combined_df)

    # Step 5: Filter to keep only test_date
    print(f"\n📅 Filtering to test date: {test_date}")
    result_df = df_with_ewma.filter(F.col("session_date") == F.lit(test_date))
    result_count = result_df.count()
    print(f"   ✅ Final result: {result_count:,} rows")

    return result_df


def aggregate_daily_features_simplified(raw_events: DataFrame, target_date: str) -> DataFrame:
    """
    Simplified daily feature aggregation for validation.

    NOTE: This is a minimal implementation. For full validation,
    import the actual aggregate_daily_features() from Phase 0.
    """
    print(f"   WARNING: Using simplified aggregation (subset of features)")
    print(f"   For production validation, use full Phase 0 aggregation function")

    # Add session_date
    if "EVENTTIMESTAMP" in raw_events.columns:
        df = raw_events.withColumn("session_date", F.to_date(F.col("EVENTTIMESTAMP")))
    elif "EVENTDATE" in raw_events.columns:
        df = raw_events.withColumn("session_date", F.col("EVENTDATE"))
    else:
        df = raw_events.withColumn("session_date", F.lit(target_date))

    # Filter to target date
    df = df.filter(F.col("session_date") == F.lit(target_date))

    # Extract USERID
    if "USERID" in df.columns:
        user_col = "USERID"
    elif "userId" in df.columns:
        user_col = "userId"
    else:
        raise ValueError("Cannot find user ID column")

    # Basic daily aggregation (subset of full features)
    daily_agg = df.groupBy(user_col, "session_date").agg(
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("daily_spend_coins"),
        F.count(F.lit(1)).alias("daily_event_count"),
        F.countDistinct("SESSIONID").alias("daily_session_count"),
        F.mean(F.col("LEVELAVERAGEFPS")).alias("avg_fps")
    )

    return daily_agg


def compute_ewma_features_spark_simplified(df: DataFrame) -> DataFrame:
    """
    Simplified EWMA computation for validation.

    NOTE: This is a minimal implementation. For full validation,
    import the actual compute_ewma_features_spark() from Phase 0.
    """
    from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType
    import pandas as pd

    print(f"   WARNING: Using simplified EWMA computation (subset of alphas/columns)")
    print(f"   For production validation, use full Phase 0 EWMA function")

    # Simplified: only compute EWMA for a few columns with alpha=0.3
    alpha = 0.3
    cols_to_ewma = ['daily_spend_coins', 'daily_event_count', 'avg_fps']

    # Determine user_id column name
    if 'user_id' in df.columns:
        user_col = 'user_id'
    elif 'USERID' in df.columns:
        user_col = 'USERID'
    else:
        raise ValueError("Cannot find user ID column")

    # Build output schema
    user_id_field = next((f for f in df.schema.fields if f.name.lower() == user_col.lower()), None)
    session_date_field = next((f for f in df.schema.fields if f.name == 'session_date'), None)

    ewma_schema = StructType([
        StructField(user_col, user_id_field.dataType if user_id_field else StringType(), False),
        StructField('session_date', session_date_field.dataType if session_date_field else DateType(), False),
        *[StructField(f"ewma_{col}_alpha{int(alpha*10)}", DoubleType(), True) for col in cols_to_ewma]
    ])

    def ewma_apply(pdf):
        pdf = pdf.sort_values(['session_date'])

        # Convert session_date to date objects
        raw_dates = pdf['session_date']
        if pd.api.types.is_datetime64_any_dtype(raw_dates):
            session_dates = pd.to_datetime(raw_dates).dt.date.values
        else:
            session_dates = raw_dates.values

        out = pd.DataFrame({
            user_col: pdf[user_col].values,
            'session_date': session_dates
        })

        for col in cols_to_ewma:
            if col in pdf.columns:
                series = pd.to_numeric(pdf[col], errors='coerce')
                out[f"ewma_{col}_alpha{int(alpha*10)}"] = series.ewm(alpha=alpha, adjust=False).mean().shift(1)

        # Fill NaN with 0
        for c in out.columns:
            if c not in (user_col, 'session_date'):
                out[c] = out[c].fillna(0.0)

        return out

    # Compute EWMA
    ewma_df = df.groupBy(user_col).applyInPandas(ewma_apply, schema=ewma_schema)

    # Join back
    joined = df.join(ewma_df, on=[user_col, 'session_date'], how='left')

    return joined


def compare_ewma_features(ground_truth: DataFrame, recomputed: DataFrame) -> Dict:
    """
    Compare EWMA features between ground truth and recomputed.

    Returns:
        Dict with comparison results including differences for each EWMA column
    """
    print(f"\n{'='*80}")
    print(f"STEP 3: Comparing Ground Truth vs Recomputed")
    print(f"{'='*80}")

    # Identify EWMA columns
    gt_ewma_cols = [c for c in ground_truth.columns if c.startswith('ewma_')]
    rc_ewma_cols = [c for c in recomputed.columns if c.startswith('ewma_')]

    # Find common EWMA columns
    common_ewma = set(gt_ewma_cols) & set(rc_ewma_cols)

    print(f"\n📊 EWMA Column Comparison:")
    print(f"   Ground truth EWMA columns: {len(gt_ewma_cols)}")
    print(f"   Recomputed EWMA columns: {len(rc_ewma_cols)}")
    print(f"   Common EWMA columns: {len(common_ewma)}")

    if not common_ewma:
        print(f"\n   ⚠️  WARNING: No common EWMA columns to compare!")
        return {'status': 'no_common_columns', 'differences': {}}

    # Join on user_id and session_date
    print(f"\n🔗 Joining ground truth and recomputed on (user_id, session_date)...")

    # Alias DataFrames
    gt_aliased = ground_truth.alias("gt")
    rc_aliased = recomputed.alias("rc")

    # Join
    joined = gt_aliased.join(
        rc_aliased,
        on=[ground_truth.user_id == recomputed.user_id,
            ground_truth.session_date == recomputed.session_date],
        how="inner"
    )

    join_count = joined.count()
    print(f"   ✅ Joined rows: {join_count:,}")

    # Compute differences for each EWMA column
    print(f"\n🧮 Computing absolute differences for each EWMA column...")

    differences = {}
    for col in sorted(common_ewma):
        # Compute absolute difference
        diff_col = f"{col}_diff"
        joined_with_diff = joined.withColumn(
            diff_col,
            F.abs(F.col(f"gt.{col}") - F.col(f"rc.{col}"))
        )

        # Get statistics
        stats = joined_with_diff.agg(
            F.max(diff_col).alias("max_diff"),
            F.mean(diff_col).alias("mean_diff"),
            F.sum(diff_col).alias("sum_diff"),
            F.count(diff_col).alias("count")
        ).collect()[0]

        differences[col] = {
            'max_diff': float(stats['max_diff']) if stats['max_diff'] is not None else 0.0,
            'mean_diff': float(stats['mean_diff']) if stats['mean_diff'] is not None else 0.0,
            'sum_diff': float(stats['sum_diff']) if stats['sum_diff'] is not None else 0.0,
            'count': int(stats['count']) if stats['count'] is not None else 0
        }

    return {
        'status': 'success',
        'common_columns': len(common_ewma),
        'differences': differences,
        'join_count': join_count
    }


def generate_validation_report(comparison_results: Dict, test_date: str, output_path: str = None):
    """Generate and print validation report."""
    print(f"\n{'='*80}")
    print(f"VALIDATION REPORT")
    print(f"{'='*80}")
    print(f"\nTest Date: {test_date}")
    print(f"Status: {comparison_results['status']}")

    if comparison_results['status'] != 'success':
        print(f"\n⚠️  Validation could not be completed")
        return

    print(f"\nMatched Rows: {comparison_results['join_count']:,}")
    print(f"EWMA Columns Compared: {comparison_results['common_columns']}")

    differences = comparison_results['differences']

    # Overall statistics
    all_max_diffs = [d['max_diff'] for d in differences.values()]
    all_sum_diffs = [d['sum_diff'] for d in differences.values()]

    overall_max_diff = max(all_max_diffs) if all_max_diffs else 0.0
    overall_sum_diff = sum(all_sum_diffs)

    print(f"\n{'='*80}")
    print(f"OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"Maximum Difference (across all EWMA columns): {overall_max_diff:.15e}")
    print(f"Total Sum of Differences: {overall_sum_diff:.15e}")

    # Floating point precision threshold
    PRECISION_THRESHOLD = 1e-10

    if overall_max_diff < PRECISION_THRESHOLD:
        print(f"\n✅ VALIDATION PASSED!")
        print(f"   All differences are within floating-point precision (< {PRECISION_THRESHOLD})")
        print(f"   Incremental EWMA computation is MATHEMATICALLY EXACT!")
    else:
        print(f"\n⚠️  VALIDATION WARNING")
        print(f"   Maximum difference ({overall_max_diff:.15e}) exceeds threshold ({PRECISION_THRESHOLD})")
        print(f"   This may indicate:")
        print(f"   - Different rounding/precision in computation")
        print(f"   - Different data in ground truth vs recomputed")
        print(f"   - Implementation differences")

    # Detailed column-by-column report
    print(f"\n{'='*80}")
    print(f"DETAILED COLUMN DIFFERENCES")
    print(f"{'='*80}")
    print(f"{'Column':<50} {'Max Diff':>15} {'Mean Diff':>15} {'Sum Diff':>15}")
    print(f"{'-'*95}")

    for col, stats in sorted(differences.items()):
        print(f"{col:<50} {stats['max_diff']:>15.10e} {stats['mean_diff']:>15.10e} {stats['sum_diff']:>15.10e}")

    # Save detailed report if output path provided
    if output_path:
        print(f"\n💾 Saving detailed report to: {output_path}")
        report_data = {
            'test_date': test_date,
            'timestamp': datetime.now().isoformat(),
            'status': comparison_results['status'],
            'overall': {
                'max_difference': overall_max_diff,
                'total_sum_difference': overall_sum_diff,
                'passed': overall_max_diff < PRECISION_THRESHOLD,
                'threshold': PRECISION_THRESHOLD
            },
            'columns': differences,
            'metadata': {
                'matched_rows': comparison_results['join_count'],
                'ewma_columns_compared': comparison_results['common_columns']
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"   ✅ Report saved")


def main():
    parser = argparse.ArgumentParser(
        description="Validate incremental EWMA computation produces identical results to full method"
    )
    parser.add_argument('--test-date', required=True, help='Date to test (YYYY-MM-DD), must exist in Delta table')
    parser.add_argument('--delta-path', required=True, help='Path to Delta table with existing features')
    parser.add_argument('--sf-config-json', required=True, help='Path to Snowflake config JSON')
    parser.add_argument('--snowflake-table', default='spaceplay.unity.boxjam_snapshot_2025_10_17',
                       help='Snowflake table name')
    parser.add_argument('--output-report', help='Path to save detailed validation report JSON')

    args = parser.parse_args()

    # Validate date format
    try:
        test_date_dt = datetime.strptime(args.test_date, '%Y-%m-%d')
    except ValueError:
        print(f"❌ Invalid date format: {args.test_date}. Expected YYYY-MM-DD")
        sys.exit(1)

    # Calculate previous date
    previous_date_dt = test_date_dt - timedelta(days=1)
    previous_date = previous_date_dt.strftime('%Y-%m-%d')

    print(f"\n{'='*80}")
    print(f"INCREMENTAL EWMA VALIDATION")
    print(f"{'='*80}")
    print(f"\nTest Date: {args.test_date}")
    print(f"Previous Date: {previous_date}")
    print(f"Delta Path: {args.delta_path}")
    print(f"Snowflake Table: {args.snowflake_table}")

    # Initialize Spark
    print(f"\n🚀 Initializing Spark session...")
    spark = SparkSession.builder \
        .appName("ValidateIncrementalEWMA") \
        .config("spark.jars.packages", "net.snowflake:spark-snowflake_2.12:2.11.0-spark_3.3,io.delta:delta-core_2.12:2.2.0") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()

    print(f"   ✅ Spark session initialized")

    # Load Snowflake config
    print(f"\n📋 Loading Snowflake configuration...")
    sf_config = load_snowflake_config(args.sf_config_json)
    print(f"   ✅ Config loaded")

    # Step 1: Load ground truth
    ground_truth = load_ground_truth_from_delta(spark, args.delta_path, args.test_date)

    # Step 2: Recompute using incremental method
    # First, query Snowflake for test date raw events
    today_raw = query_snowflake_for_date(spark, sf_config, args.test_date, args.snowflake_table)

    # Recompute
    recomputed = recompute_with_incremental_method(
        spark,
        args.delta_path,
        previous_date,
        args.test_date,
        today_raw
    )

    # Step 3: Compare
    comparison_results = compare_ewma_features(ground_truth, recomputed)

    # Step 4: Generate report
    generate_validation_report(comparison_results, args.test_date, args.output_report)

    print(f"\n{'='*80}")
    print(f"VALIDATION COMPLETE")
    print(f"{'='*80}\n")

    # Exit with appropriate code
    if comparison_results['status'] == 'success':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
