#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bandit_datagen_daily_pure_spark.py
==================================

Pure Spark version of the daily-level contextual bandit pipeline for SpacePlay
difficulty selection. All feature engineering is implemented with Spark
DataFrame operations and window functions (no pandas transformations).

Key differences vs bandit_datagen_daily_fast.py:
- Removes any toPandas() conversion and pandas-based feature engineering
- Implements advanced gameplay features using Spark window functions
- Saves Parquet outputs partitioned by user and date for efficient loading

Outputs (Parquet):
- daily_features_spark.parquet/ (core + advanced features)

Author: Codex (pure Spark translation)
Date: 2025-10-27
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import shutil
from typing import List, Tuple, Optional
from datetime import datetime

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType, LongType, StringType,
    FloatType, DecimalType, DateType, DataType
)


def initialize_spark_session(shuffle_partitions: int = 2000,
                             local_dirs: str = "",
                             enable_aqe: bool = True,
                             extra_jars: str = "",
                             extra_packages: str = "",
                             enable_delta_extensions: bool = False) -> SparkSession:
    """Initialize Spark session; tune shuffles, AQE, and local dirs."""
    print("⚡ Initializing Spark session (Pure Spark pipeline)...")

    jar_path = "/Users/yohanmedalsy/Desktop/Personal_Projects/spark_jars/snowflake-jdbc-3.14.4.jar"

    # Ensure driver and executor use the same Python version
    import sys
    python_executable = sys.executable
    
    builder = (
        SparkSession.builder
        .appName("SpacePlay_Daily_Bandit_Pipeline_PURE_SPARK")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.master", "local[*]")
        .config("spark.driver.memory", "110g")
        .config("spark.driver.maxResultSize", "50g")
        .config("spark.executor.memory", "110g")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.jars", jar_path)
        .config("spark.pyspark.python", python_executable)  # Set executor Python to match driver
        .config("spark.pyspark.driver.python", python_executable)  # Set driver Python explicitly
        .config("spark.default.parallelism", str(shuffle_partitions))  # Set default parallelism to match shuffle partitions
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", str(enable_aqe).lower())
        .config("spark.sql.adaptive.coalescePartitions.enabled", str(enable_aqe).lower())
        .config("spark.sql.adaptive.skewJoin.enabled", str(enable_aqe).lower())
    )

    if local_dirs:
        builder = builder.config("spark.local.dir", local_dirs)
        # Additional shuffle reliability configs when using custom local dirs
        # Spark 4.0: use spark.shuffle.localDisk.file.output.buffer instead of deprecated spark.shuffle.unsafe.file.output.buffer
        builder = builder \
            .config("spark.shuffle.localDisk.file.output.buffer", "512k") \
            .config("spark.shuffle.file.buffer", "512k") \
            .config("spark.shuffle.registration.timeout", "600000") \
            .config("spark.shuffle.registration.maxAttempts", "5")
    else:
        # When no custom local_dirs, ensure default temp directory stability
        builder = builder \
            .config("spark.shuffle.localDisk.file.output.buffer", "256k") \
            .config("spark.shuffle.file.buffer", "256k")
    if extra_jars:
        current_jars = jar_path if jar_path else ""
        combined = f"{current_jars},{extra_jars}" if current_jars and extra_jars else (extra_jars or current_jars)
        builder = builder.config("spark.jars", combined)
    if extra_packages:
        builder = builder.config("spark.jars.packages", extra_packages)
    if enable_delta_extensions:
        builder = builder \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

    spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("✅ Spark session initialized")
    print(f"   AQE: {enable_aqe}, shuffle partitions: {shuffle_partitions}")
    if local_dirs:
        print(f"   Local shuffle dirs: {local_dirs}")
    return spark


def connect_to_snowflake_direct() -> Tuple[dict, str]:
    """Return Snowflake connection parameters and JDBC URL."""
    print("🔗 Preparing Snowflake connection parameters...")

    snowflake_config = {
        "user": "YOUR_SNOWFLAKE_USER",
        "password": "YOUR_SNOWFLAKE_PASSWORD",
        "account": "YOUR_SNOWFLAKE_ACCOUNT",
        "warehouse": "DASHBOARD_DEFAULT",
        "database": "SPACEPLAY",
        "schema": "UNITY",
    }

    jdbc_url = f"jdbc:snowflake://{snowflake_config['account']}.snowflakecomputing.com/"

    print("✅ Snowflake configuration ready")
    return snowflake_config, jdbc_url


def load_snowflake_config_from_json(path: str) -> Optional[dict]:
    """Load Snowflake configuration from a JSON file if it exists.

    Expected keys (any subset is fine):
      url | account, user, password, warehouse, database, schema, role

    Returns None if the file is missing or malformed.
    """
    try:
        if not path:
            return None
        if not os.path.exists(path):
            print(f"ℹ️  Snowflake JSON config not found at: {path} (skipping)")
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"⚠️  Snowflake JSON config at {path} is not an object; ignoring")
            return None
        # Normalize keys to upper/expected names
        norm = {k.lower(): v for k, v in data.items()}
        cfg = {
            'url': norm.get('url') or norm.get('sf_url') or norm.get('sfurl'),
            'account': norm.get('account') or norm.get('sf_account') or norm.get('sfaccount'),
            'user': norm.get('user') or norm.get('sf_user') or norm.get('sfuser'),
            'password': norm.get('password') or norm.get('sf_password') or norm.get('sfpassword'),
            'warehouse': norm.get('warehouse') or norm.get('sf_warehouse') or norm.get('sfwarehouse'),
            'database': norm.get('database') or norm.get('sf_database') or norm.get('sfdatabase'),
            'schema': norm.get('schema') or norm.get('sf_schema') or norm.get('sfschema'),
            'role': norm.get('role') or norm.get('sf_role') or norm.get('sfrole'),
        }
        return cfg
    except Exception as e:
        print(f"⚠️  Failed to load Snowflake JSON config from {path}: {e}")
        return None


def create_spark_dataframe(spark: SparkSession,
                           snowflake_config: dict,
                           jdbc_url: str,
                           start_date: str = "2025-07-01") -> DataFrame:
    """Load raw SpacePlay events from Snowflake into Spark via JDBC."""
    print("📥 Loading SpacePlay events from Snowflake...")
    print(f"   Date filter: gauserstartdate >= '{start_date}'")

    query = f"""
    SELECT *,
           MOD(ABS(HASH(USERID)::NUMBER(38,0)), 20) AS partition_key
    FROM spaceplay.unity.boxjam_snapshot_2025_10_17
    WHERE gameName = 'Box Jam'
      AND environmentName = 'Live'
      AND gauserstartdate >= '{start_date}'
    """

    print("   Using 20 partitions for parallel loading (hash partitioning)...")
    df = (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", f"({query}) AS subquery")
        .option("user", snowflake_config["user"])
        .option("password", snowflake_config["password"])
        .option("driver", "net.snowflake.client.jdbc.SnowflakeDriver")
        .option("partitionColumn", "partition_key")
        .option("lowerBound", "0")
        .option("upperBound", "19")
        .option("numPartitions", "20")
        .option("fetchsize", "10000")
        .load()
        .drop("partition_key")
    )

    print("✅ Snowflake data loaded (row count skipped for speed)")
    return df


def create_snowflake_connector_dataframe(spark: SparkSession,
                                         sf_options: dict,
                                         start_date: str = "2025-07-01") -> DataFrame:
    """Load SpacePlay events using the Spark Snowflake Connector."""
    print("📥 Loading SpacePlay events via Spark Snowflake Connector...")
    print(f"   Date filter: gauserstartdate >= '{start_date}'")

    query = f"""
    SELECT *
    FROM spaceplay.unity.boxjam_snapshot_2025_10_17
    WHERE gameName = 'Box Jam'
      AND environmentName = 'Live'
      AND gauserstartdate >= '{start_date}'
    """

    df = (
        spark.read
             .format("net.snowflake.spark.snowflake")
             .options(**sf_options)
             .option("query", query)
             .load()
    )
    print("✅ Snowflake Connector load complete (row count skipped)")
    return df


def load_from_parquet(spark: SparkSession, parquet_path: str) -> DataFrame:
    """Load raw data from Parquet export."""
    print(f"📂 Loading SpacePlay events from Parquet: {parquet_path}")
    df = spark.read.parquet(parquet_path)
    if "partition_key" in df.columns:
        df = df.drop("partition_key")
    print("✅ Parquet data loaded (row count skipped for speed)")
    return df


def load_from_csv(spark: SparkSession, csv_path: str) -> DataFrame:
    """Load raw data from CSV export directory."""
    print(f"📂 Loading SpacePlay events from CSV: {csv_path}")
    df = spark.read.csv(csv_path, header=True, inferSchema=False)
    if "partition_key" in df.columns:
        df = df.drop("partition_key")
    print("✅ CSV data loaded (row count skipped for speed)")
    return df


def add_session_date_column(events_df: DataFrame) -> DataFrame:
    """Add session_date from timestamp columns for daily grouping."""
    print("🗓️ Adding session_date column from EVENTTIMESTAMP/EVENTDATE...")
    if "EVENTTIMESTAMP" in events_df.columns:
        return events_df.withColumn("session_date", F.to_date(F.col("EVENTTIMESTAMP")))
    if "EVENTDATE" in events_df.columns:
        return events_df.withColumn("session_date", F.col("EVENTDATE"))
    print("⚠️ No timestamp column found; defaulting to '2023-01-01'")
    return events_df.withColumn("session_date", F.lit("2023-01-01"))


def retain_multi_day_users(events_df: DataFrame,
                           user_col: str = "USERID",
                           date_col: str = "session_date") -> DataFrame:
    """Keep only users with activity on ≥2 distinct days."""
    print("🔍 Filtering to users with activity on ≥2 different days...")
    days_per_user = events_df.groupBy(user_col).agg(F.countDistinct(date_col).alias("distinct_session_days"))
    multi_day_users = days_per_user.filter(F.col("distinct_session_days") >= 2).select(user_col)
    filtered_df = events_df.join(multi_day_users, on=user_col, how="inner")
    print("✅ Multi-day user filtering complete (counts skipped)")
    return filtered_df


def extract_logged_action(daily_events: DataFrame) -> DataFrame:
    """Extract first LEVELEFFECTIVEMULTIPLIER from 'levelStarted' per user-day (assist=0)."""
    print("🎯 Extracting logged action from first levelStarted per user-day...")
    level_started = daily_events.filter(
        (F.col("EVENTNAME") == "levelStarted") &
        (F.col("LEVELEFFECTIVEMULTIPLIER").isNotNull()) &
        (F.col("LEVELDIFFICULTYDDAASSIST") == 0)
    )
    w = Window.partitionBy("USERID", "session_date").orderBy("EVENTTIMESTAMP")
    logged_action = (
        level_started
        .withColumn("row_num", F.row_number().over(w))
        .filter(F.col("row_num") == 1)
        .select(
            "USERID",
            "session_date",
            F.col("LEVELEFFECTIVEMULTIPLIER").alias("logged_difficulty_multiplier")
        )
    )
    print("✅ Logged action extracted")
    return logged_action


def assign_difficulty_arms(df_with_action: DataFrame) -> DataFrame:
    """Assign 5 difficulty arms based on constrained action deltas (lagged next-day)."""
    print("🎲 Computing action deltas and assigning 5-arm difficulty labels...")
    w_user_day = Window.partitionBy("USERID").orderBy("session_date")

    df_with_next = df_with_action.withColumn(
        "next_day_multiplier_actual", F.lead("logged_difficulty_multiplier", 1).over(w_user_day)
    )

    df_with_action_col = df_with_next.withColumn(
        "action",
        F.round(F.col("next_day_multiplier_actual") - F.col("logged_difficulty_multiplier"), 2)
    ).filter(F.col("next_day_multiplier_actual").isNotNull())

    # One-sided directional constraints (bounds [0.5, 1.25])
    df_with_action_col = df_with_action_col.withColumn(
        "action",
        F.when((F.col("logged_difficulty_multiplier") <= 0.5) & (F.col("action") < 0), 0.00)
         .when((F.col("logged_difficulty_multiplier") >= 1.25) & (F.col("action") > 0), 0.00)
         .otherwise(F.col("action"))
    )

    df_with_action_col = df_with_action_col.withColumn(
        "next_effectivelevelmultiplier",
        F.round(F.col("logged_difficulty_multiplier") + F.col("action"), 2)
    )

    df_with_arms = df_with_action_col.withColumn(
        "difficulty_arm",
        F.when(F.col("action") == 0.00, "Same")
         .when((F.col("action") >= 0.05) & (F.col("action") <= 0.07), "Harder")
         .when((F.col("action") >= 0.11) & (F.col("action") <= 0.13), "Harderer")
         .when((F.col("action") >= -0.07) & (F.col("action") <= -0.05), "Easier")
         .when((F.col("action") >= -0.13) & (F.col("action") <= -0.11), "Easierer")
         .otherwise(None)
    )

    df_filtered = df_with_arms.filter(F.col("difficulty_arm").isNotNull()).drop("next_day_multiplier_actual")
    print("✅ Arm assignment complete")
    return df_filtered


def aggregate_daily_features(daily_events: DataFrame) -> DataFrame:
    """Aggregate raw events to one row per user-day with core features."""
    print("📊 Aggregating daily features (user-day level)...")

    # First event timestamp per user (for days_since_install)
    user_first_event = daily_events.groupBy("USERID").agg(
        F.min("EVENTTIMESTAMP").alias("first_event_timestamp")
    )
    daily_events = daily_events.join(user_first_event, on="USERID", how="left")

    # Basic daily aggregation
    daily_basic = daily_events.groupBy("USERID", "session_date").agg(
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("current_spend"),
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("current_exchangespentamountcoins"),

        # Real currency stats
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("daily_spend_real_sum"),
        F.mean(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_real_mean"),
        F.stddev(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_real_std"),
        F.max(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_real_max"),
        F.min(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_real_min"),

        # Virtual currency stats
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("daily_spend_coins_sum"),
        F.mean(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_coins_mean"),
        F.stddev(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_coins_std"),
        F.max(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_coins_max"),
        F.min(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_coins_min"),

        # Legacy compatibility
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("daily_spend_real"),
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("daily_spend_coins"),

        # Timestamps
        F.min("EVENTTIMESTAMP").alias("day_start_ts"),
        F.max("EVENTTIMESTAMP").alias("day_end_ts"),

        # User context (first/last of day)
        F.first("USERSKILLSEGMENT", ignorenulls=True).alias("user_segment"),
        F.first("PLATFORM", ignorenulls=True).alias("platform"),
        F.first("USERLEVEL", ignorenulls=True).alias("user_level_start"),
        F.last("USERLEVEL", ignorenulls=True).alias("user_level_end"),

        # Activity metrics
        F.count(F.lit(1)).alias("daily_event_count"),
        F.countDistinct("SESSIONID").alias("daily_session_count")
    )

    # Level performance metrics
    print("📊 Calculating daily level performance metrics...")
    level_events = daily_events.filter(
        F.col("EVENTNAME").isin(["levelStarted", "levelEnded", "levelOutOfSpace", "levelOutOfTime"]) 
    )
    level_features = level_events.groupBy("USERID", "session_date").agg(
        F.count(F.when(F.col("EVENTNAME") == "levelStarted", 1)).alias("levels_started"),
        F.count(F.when(
            F.col("EVENTNAME").isin(["levelEnded", "levelOutOfSpace", "levelOutOfTime"]) &
            (F.col("LEVELRESULT") == "Victory"), 1
        )).alias("levels_completed"),
        F.count(F.when(
            F.col("EVENTNAME").isin(["levelEnded", "levelOutOfSpace", "levelOutOfTime"]) &
            F.col("LEVELRESULT").isin(["Defeat", "Abort"]), 1
        )).alias("levels_failed"),

        # Attempts
        F.sum(F.when(F.col("LEVELATTEMPTS").isNotNull(), F.col("LEVELATTEMPTS")).otherwise(0)).alias("total_level_attempts"),
        F.mean(F.when(F.col("LEVELATTEMPTS").isNotNull(), F.col("LEVELATTEMPTS"))).alias("avg_level_attempts"),
        F.stddev(F.col("LEVELATTEMPTS")).alias("level_attempts_std"),
        F.max(F.col("LEVELATTEMPTS")).alias("level_attempts_max"),
        F.min(F.col("LEVELATTEMPTS")).alias("level_attempts_min"),

        F.countDistinct("LEVELINDEX").alias("distinct_levels_played"),

        # Resurrections (desperation signals - comprehensive statistics)
        F.sum(F.coalesce(F.col("LEVELOUTOFSPACERESURRECTIONS"), F.lit(0))).alias("resurrections_space_sum"),
        F.mean(F.col("LEVELOUTOFSPACERESURRECTIONS")).alias("resurrections_space_mean"),
        F.stddev(F.col("LEVELOUTOFSPACERESURRECTIONS")).alias("resurrections_space_std"),
        F.max(F.col("LEVELOUTOFSPACERESURRECTIONS")).alias("resurrections_space_max"),

        F.sum(F.coalesce(F.col("LEVELTIMEUPRESURRECTIONS"), F.lit(0))).alias("resurrections_time_sum"),
        F.mean(F.col("LEVELTIMEUPRESURRECTIONS")).alias("resurrections_time_mean"),
        F.stddev(F.col("LEVELTIMEUPRESURRECTIONS")).alias("resurrections_time_std"),
        F.max(F.col("LEVELTIMEUPRESURRECTIONS")).alias("resurrections_time_max"),

        # Legacy compatibility
        F.sum(F.coalesce(F.col("LEVELOUTOFSPACERESURRECTIONS"), F.lit(0))).alias("resurrections_space"),
        F.sum(F.coalesce(F.col("LEVELTIMEUPRESURRECTIONS"), F.lit(0))).alias("resurrections_time"),

        # Time metrics (comprehensive statistics)
        F.sum(F.coalesce(F.col("LEVELDURATION"), F.lit(0))).alias("total_level_duration"),
        F.mean(F.col("LEVELDURATION")).alias("avg_level_duration"),
        F.stddev(F.col("LEVELDURATION")).alias("level_duration_std"),
        F.max(F.col("LEVELDURATION")).alias("level_duration_max"),
        F.min(F.col("LEVELDURATION")).alias("level_duration_min"),

        F.mean(F.col("LEVELTIMEREMAINING")).alias("avg_time_remaining"),
        F.stddev(F.col("LEVELTIMEREMAINING")).alias("time_remaining_std"),
        F.max(F.col("LEVELTIMEREMAINING")).alias("time_remaining_max"),
        F.min(F.col("LEVELTIMEREMAINING")).alias("time_remaining_min"),

        # Booster usage (comprehensive statistics for each type)
        # Undo boosters
        F.sum(F.coalesce(F.col("LEVELUNDOBOOSTERUSED"), F.lit(0))).alias("undo_boosters_sum"),
        F.mean(F.col("LEVELUNDOBOOSTERUSED")).alias("undo_boosters_mean"),
        F.stddev(F.col("LEVELUNDOBOOSTERUSED")).alias("undo_boosters_std"),
        F.max(F.col("LEVELUNDOBOOSTERUSED")).alias("undo_boosters_max"),

        # Extrabox boosters
        F.sum(F.coalesce(F.col("LEVELEXTRABOXBOOSTERUSED"), F.lit(0))).alias("extrabox_boosters_sum"),
        F.mean(F.col("LEVELEXTRABOXBOOSTERUSED")).alias("extrabox_boosters_mean"),
        F.stddev(F.col("LEVELEXTRABOXBOOSTERUSED")).alias("extrabox_boosters_std"),
        F.max(F.col("LEVELEXTRABOXBOOSTERUSED")).alias("extrabox_boosters_max"),

        # Hammer boosters
        F.sum(F.coalesce(F.col("LEVELHAMMERBOOSTERUSED"), F.lit(0))).alias("hammer_boosters_sum"),
        F.mean(F.col("LEVELHAMMERBOOSTERUSED")).alias("hammer_boosters_mean"),
        F.stddev(F.col("LEVELHAMMERBOOSTERUSED")).alias("hammer_boosters_std"),
        F.max(F.col("LEVELHAMMERBOOSTERUSED")).alias("hammer_boosters_max"),

        # Joker boosters
        F.sum(F.coalesce(F.col("LEVELJOKERBOOSTERUSED"), F.lit(0))).alias("joker_boosters_sum"),
        F.mean(F.col("LEVELJOKERBOOSTERUSED")).alias("joker_boosters_mean"),
        F.stddev(F.col("LEVELJOKERBOOSTERUSED")).alias("joker_boosters_std"),
        F.max(F.col("LEVELJOKERBOOSTERUSED")).alias("joker_boosters_max"),

        # PreJoker boosters
        F.sum(F.coalesce(F.col("LEVELPREJOKERBOOSTERUSED"), F.lit(0))).alias("prejoker_boosters_sum"),
        F.mean(F.col("LEVELPREJOKERBOOSTERUSED")).alias("prejoker_boosters_mean"),
        F.stddev(F.col("LEVELPREJOKERBOOSTERUSED")).alias("prejoker_boosters_std"),
        F.max(F.col("LEVELPREJOKERBOOSTERUSED")).alias("prejoker_boosters_max"),

        # Supersuit boosters
        F.sum(F.coalesce(F.col("LEVELSUPERSUITBOOSTERUSED"), F.lit(0))).alias("supersuit_boosters_sum"),
        F.mean(F.col("LEVELSUPERSUITBOOSTERUSED")).alias("supersuit_boosters_mean"),
        F.stddev(F.col("LEVELSUPERSUITBOOSTERUSED")).alias("supersuit_boosters_std"),
        F.max(F.col("LEVELSUPERSUITBOOSTERUSED")).alias("supersuit_boosters_max"),

        # Legacy names for backward compatibility
        F.sum(F.coalesce(F.col("LEVELUNDOBOOSTERUSED"), F.lit(0))).alias("undo_boosters"),
        F.sum(F.coalesce(F.col("LEVELEXTRABOXBOOSTERUSED"), F.lit(0))).alias("extrabox_boosters"),
        F.sum(F.coalesce(F.col("LEVELHAMMERBOOSTERUSED"), F.lit(0))).alias("hammer_boosters"),
        F.sum(F.coalesce(F.col("LEVELJOKERBOOSTERUSED"), F.lit(0))).alias("joker_boosters"),
        F.sum(F.coalesce(F.col("LEVELPREJOKERBOOSTERUSED"), F.lit(0))).alias("prejoker_boosters"),
        F.sum(F.coalesce(F.col("LEVELSUPERSUITBOOSTERUSED"), F.lit(0))).alias("supersuit_boosters"),

        # Difficulty and performance (comprehensive statistics)
        F.mean(F.col("LEVELRANKEASING")).alias("avg_rank_easing"),
        F.stddev(F.col("LEVELRANKEASING")).alias("rank_easing_std"),
        F.max(F.col("LEVELRANKEASING")).alias("rank_easing_max"),
        F.min(F.col("LEVELRANKEASING")).alias("rank_easing_min"),

        F.mean(F.col("LEVELEFFECTIVEMULTIPLIER")).alias("avg_effective_multiplier"),
        F.stddev(F.col("LEVELEFFECTIVEMULTIPLIER")).alias("effective_multiplier_std"),
        F.max(F.col("LEVELEFFECTIVEMULTIPLIER")).alias("effective_multiplier_max"),
        F.min(F.col("LEVELEFFECTIVEMULTIPLIER")).alias("effective_multiplier_min"),

        F.mean(F.col("LEVELAVERAGEFPS")).alias("avg_fps"),
        F.stddev(F.col("LEVELAVERAGEFPS")).alias("fps_std"),
        F.max(F.col("LEVELAVERAGEFPS")).alias("fps_max"),
        F.min(F.col("LEVELAVERAGEFPS")).alias("fps_min")
    )

    # Economy & user state features
    print("📊 Adding daily economy and user state features...")
    economy_features = daily_events.groupBy("USERID", "session_date").agg(
        F.datediff(F.first("session_date"), F.first("first_event_timestamp")).alias("days_since_install"),

        F.first("USERCOINBALANCE", ignorenulls=True).alias("coin_balance_start"),
        F.last("USERCOINBALANCE", ignorenulls=True).alias("coin_balance_end"),

        F.first("USERLIFEBALANCE", ignorenulls=True).alias("life_balance_start"),
        F.last("USERLIFEBALANCE", ignorenulls=True).alias("life_balance_end"),

        F.first("USERCROWNCOUNT", ignorenulls=True).alias("crown_count_start")
            if "USERCROWNCOUNT" in daily_events.columns else F.lit(None).alias("crown_count_start"),
        F.last("USERCROWNCOUNT", ignorenulls=True).alias("crown_count_end")
            if "USERCROWNCOUNT" in daily_events.columns else F.lit(None).alias("crown_count_end"),

        F.first("USERUNDOBOOSTERBALANCE", ignorenulls=True).alias("undo_balance_start")
            if "USERUNDOBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("undo_balance_start"),
        F.last("USERUNDOBOOSTERBALANCE", ignorenulls=True).alias("undo_balance_end")
            if "USERUNDOBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("undo_balance_end"),
        F.first("USERHAMMERBOOSTERBALANCE", ignorenulls=True).alias("hammer_balance_start")
            if "USERHAMMERBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("hammer_balance_start"),
        F.last("USERHAMMERBOOSTERBALANCE", ignorenulls=True).alias("hammer_balance_end")
            if "USERHAMMERBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("hammer_balance_end"),

        F.first("USEREXTRABOXBOOSTERBALANCE", ignorenulls=True).alias("extrabox_balance_start")
            if "USEREXTRABOXBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("extrabox_balance_start"),
        F.last("USEREXTRABOXBOOSTERBALANCE", ignorenulls=True).alias("extrabox_balance_end")
            if "USEREXTRABOXBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("extrabox_balance_end"),

        F.count(F.when(F.col("EVENTNAME").like("%ad%") | F.col("EVENTNAME").like("%Ad%"), 1)).alias("ad_events"),
        F.sum(F.when(F.col("ADECPMUSD").isNotNull(), F.col("ADECPMUSD").cast("double")).otherwise(0.0)).alias("ad_revenue")
    )

    # Join
    print("📊 Joining all daily feature groups...")
    daily_aggregated = (
        daily_basic
        .join(level_features, on=["USERID", "session_date"], how="left")
        .join(economy_features, on=["USERID", "session_date"], how="left")
    )

    # Derived metrics
    print("📊 Computing derived metrics...")
    daily_aggregated = daily_aggregated \
        .withColumn(
            "current_avg_attemptperuser",
            F.when(F.col("distinct_levels_played") > 0,
                   F.col("total_level_attempts") / F.col("distinct_levels_played")).otherwise(0.0)
        ) \
        .withColumn("day_of_week", F.dayofweek(F.col("session_date"))) \
        .withColumn("day_of_month", F.dayofmonth(F.col("session_date"))) \
        .withColumn("month", F.month(F.col("session_date"))) \
        .withColumn("is_weekend", F.when(F.dayofweek(F.col("session_date")).isin([1, 7]), 1).otherwise(0)) \
        .withColumn(
            "completion_rate",
            F.when(F.col("levels_started") > 0,
                   F.col("levels_completed") / F.col("levels_started")).otherwise(0.0)
        ) \
        .withColumn(
            "total_boosters_used",
            F.coalesce(F.col("undo_boosters"), F.lit(0)) +
            F.coalesce(F.col("extrabox_boosters"), F.lit(0)) +
            F.coalesce(F.col("hammer_boosters"), F.lit(0)) +
            F.coalesce(F.col("joker_boosters"), F.lit(0)) +
            F.coalesce(F.col("prejoker_boosters"), F.lit(0)) +
            F.coalesce(F.col("supersuit_boosters"), F.lit(0))
        ) \
        .withColumn(
            "booster_usage_rate",
            F.when(F.col("levels_started") > 0,
                   F.col("total_boosters_used") / F.col("levels_started")).otherwise(0.0)
        ) \
        .withColumn(
            "total_resurrections",
            F.coalesce(F.col("resurrections_space"), F.lit(0)) +
            F.coalesce(F.col("resurrections_time"), F.lit(0))
        ) \
        .withColumn(
            "resurrection_rate",
            F.when(F.col("levels_started") > 0,
                   F.col("total_resurrections") / F.col("levels_started")).otherwise(0.0)
        ) \
        .withColumn("coin_balance_change", F.coalesce(F.col("coin_balance_end"), F.lit(0)) - F.coalesce(F.col("coin_balance_start"), F.lit(0))) \
        .withColumn("life_balance_change", F.coalesce(F.col("life_balance_end"), F.lit(0)) - F.coalesce(F.col("life_balance_start"), F.lit(0))) \
        .withColumn("user_level_gain", F.coalesce(F.col("user_level_end"), F.lit(0)) - F.coalesce(F.col("user_level_start"), F.lit(0))) \
        .withColumn("undo_balance_change", F.coalesce(F.col("undo_balance_end"), F.lit(0)) - F.coalesce(F.col("undo_balance_start"), F.lit(0))) \
        .withColumn("hammer_balance_change", F.coalesce(F.col("hammer_balance_end"), F.lit(0)) - F.coalesce(F.col("hammer_balance_start"), F.lit(0))) \
        .withColumn("extrabox_balance_change", F.coalesce(F.col("extrabox_balance_end"), F.lit(0)) - F.coalesce(F.col("extrabox_balance_start"), F.lit(0))) \
        .withColumn("daily_play_duration_seconds", F.unix_timestamp(F.col("day_end_ts")) - F.unix_timestamp(F.col("day_start_ts"))) \
        .withColumn("avg_session_duration", F.when(F.col("daily_session_count") > 0, F.col("daily_play_duration_seconds") / F.col("daily_session_count")).otherwise(0.0))

    # Fill some common nulls
    numeric_cols = [
        "current_spend", "current_exchangespentamountcoins", "levels_started",
        "levels_completed", "levels_failed", "total_boosters_used", "total_resurrections",
        "ad_events", "ad_revenue"
    ]
    daily_aggregated = daily_aggregated.fillna(0, subset=[c for c in numeric_cols if c in daily_aggregated.columns])

    print("✅ Daily feature aggregation complete")
    return daily_aggregated


def create_daily_bandit_dataset(daily_features: DataFrame,
                                logged_actions: DataFrame,
                                arm_assignments: DataFrame) -> DataFrame:
    """Join features + logged actions + assigned arms, add temporal targets/lag features."""
    print("🔗 Creating daily bandit dataset...")
    df_with_action = daily_features.join(
        logged_actions.withColumnRenamed("logged_difficulty_multiplier", "current_effectivelevelmultiplier"),
        on=["USERID", "session_date"], how="inner"
    )
    df_with_arms = df_with_action.join(
        arm_assignments.select("USERID", "session_date", "action", "next_effectivelevelmultiplier", "difficulty_arm"),
        on=["USERID", "session_date"], how="inner"
    )

    w = Window.partitionBy("USERID").orderBy("session_date")
    df_with_temporal = df_with_arms.withColumn(
        "next_exchangespentamountcoins", F.lead("current_exchangespentamountcoins", 1).over(w)
    ).withColumn(
        "next_day_reward", F.col("next_exchangespentamountcoins")
    ).withColumn(
        "previous_day_action", F.lag("action", 1).over(w)
    ).withColumn(
        "previous_day_multiplier", F.lag("current_effectivelevelmultiplier", 1).over(w)
    )

    df_final = df_with_temporal.filter(F.col("next_exchangespentamountcoins").isNotNull()) \
        .withColumn("previous_day_action", F.coalesce(F.col("previous_day_action"), F.lit(0.00))) \
        .withColumn("previous_day_multiplier", F.coalesce(F.col("previous_day_multiplier"), F.col("current_effectivelevelmultiplier")))

    print("✅ Bandit dataset created (rows with next-day reward kept)")
    return df_final


def normalize_to_user_lowercase(df: DataFrame) -> DataFrame:
    """Normalize identifier column names for downstream consistency."""
    rename_map = {}
    if "USERID" in df.columns:
        rename_map["USERID"] = "user_id"
    # Spark doesn't support bulk rename; chain with select/alias
    cols = [F.col(c).alias(rename_map.get(c, c)) for c in df.columns]
    return df.select(*cols)


def clean_null_and_zero_columns_spark(df: DataFrame, id_cols: List[str]) -> DataFrame:
    """
    Drop columns that are:
    1) all NULL, or 2) numeric and ≥99% zeros (including all zeros).
    Implemented using Spark aggregations only.
    """
    print("🧹 Cleaning null and low-variance (zeros) columns in Spark...")

    # Quick empty check (cheaper than full count)
    if df.head(1) == []:
        return df

    # Determine numeric columns
    numeric_types = {"int", "bigint", "double", "float", "decimal", "smallint", "tinyint"}
    numeric_cols = [c for c, t in df.dtypes if any(nt in t for nt in numeric_types)]
    candidates = [c for c in df.columns if c not in id_cols]

    # Null counts + total count (compute in single pass)
    null_aggs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}__nulls") for c in candidates]
    null_aggs.append(F.count(F.lit(1)).alias("__total_count"))
    null_row = df.agg(*null_aggs).collect()[0].asDict()
    total_count = int(null_row.pop("__total_count", 0))

    # Zero counts for numeric columns
    zero_aggs = [F.sum(F.when(F.col(c) == 0, 1).otherwise(0)).alias(f"{c}__zeros") for c in numeric_cols]
    zero_row = df.agg(*zero_aggs).collect()[0].asDict() if zero_aggs else {}

    drop_cols = []
    for c in candidates:
        nulls = int(null_row.get(f"{c}__nulls", 0))
        if nulls == total_count:
            drop_cols.append(c)
            continue
        if c in numeric_cols:
            zeros = int(zero_row.get(f"{c}__zeros", 0))
            zero_pct = (zeros / total_count) * 100.0
            if zero_pct >= 99.0:
                drop_cols.append(c)

    if drop_cols:
        print(f"   Dropping {len(drop_cols)} columns: {', '.join(sorted(drop_cols)[:12])}{' ...' if len(drop_cols)>12 else ''}")
        df = df.drop(*drop_cols)
    else:
        print("   No columns dropped")
    return df


def add_advanced_gameplay_features_spark(df: DataFrame) -> DataFrame:
    """Compute advanced gameplay features using Spark window functions only."""
    print("🎮 Computing advanced gameplay features (pure Spark)...")

    # We expect identifiers as user_id and session_date
    df = normalize_to_user_lowercase(df)
    original_cols = set(df.columns)

    w = Window.partitionBy("user_id").orderBy("session_date")

    def lagged_avg(col: str, k: int):
        return F.avg(F.col(col)).over(w.rowsBetween(-k, -1))

    def lagged_sum(col: str, k: int):
        return F.sum(F.col(col)).over(w.rowsBetween(-k, -1))

    def lagged_std(col: str, k: int):
        return F.stddev_samp(F.col(col)).over(w.rowsBetween(-k, -1))

    # 1) Rolling win rate (completion_rate)
    for k in [3, 5, 7]:
        df = df.withColumn(f"rolling_win_rate_{k}d", lagged_avg("completion_rate", k))

    # 2) Difficulty comfort zone
    for k in [3, 5, 7]:
        df = df.withColumn(f"difficulty_variance_{k}d", lagged_std("current_effectivelevelmultiplier", k)) \
               .withColumn(f"preferred_difficulty_{k}d", lagged_avg("current_effectivelevelmultiplier", k))

    # 3) Consecutive days played streak (lagged)
    prev_date = F.lag("session_date", 1).over(w)
    is_new_segment = F.when(prev_date.isNull() | (F.datediff(F.col("session_date"), prev_date) > 1), 1).otherwise(0)
    seg_id = F.sum(is_new_segment).over(w.rowsBetween(Window.unboundedPreceding, 0))
    df = df.withColumn("__seg_id_days__", seg_id)
    w_seg = Window.partitionBy("user_id", "__seg_id_days__").orderBy("session_date")
    streak_incl = F.row_number().over(w_seg)
    df = df.withColumn("__streak_days_curr__", streak_incl)
    df = df.withColumn("consecutive_days_streak", F.coalesce(F.lag("__streak_days_curr__", 1).over(w), F.lit(0)))

    # 4) Performance momentum: recent (last 3) - older (days -10..-4)
    recent_perf = lagged_avg("completion_rate", 3)
    older_perf = F.avg(F.col("completion_rate")).over(w.rowsBetween(-10, -4))
    df = df.withColumn("performance_momentum", recent_perf - older_perf)

    # 5) Rolling completion stats and rates
    for k in [3, 5]:
        comp = lagged_sum("levels_completed", k)
        starts = lagged_sum("levels_started", k)
        df = df.withColumn(f"rolling_completions_{k}d", comp) \
               .withColumn(f"rolling_completion_rate_{k}d", F.when(starts > 0, comp / starts).otherwise(0.0))

    # 6) Skill stability (variance of avg attempts)
    for k in [3, 5]:
        df = df.withColumn(f"skill_variance_{k}d", lagged_std("current_avg_attemptperuser", k))

    # 7) Engagement streak: high engagement days in last 3/5 (global median threshold)
    # Compute global median using approxQuantile (Spark-only)
    median_play = df.stat.approxQuantile("daily_play_duration_seconds", [0.5], 0.01)[0] if "daily_play_duration_seconds" in df.columns else None
    if median_play is not None:
        df = df.withColumn("__high_engagement__", F.when(F.col("daily_play_duration_seconds") >= F.lit(median_play), 1).otherwise(0))
        for k in [3, 5]:
            df = df.withColumn(f"high_engagement_days_{k}d", lagged_sum("__high_engagement__", k))

    # 8) Spending propensity indicators
    spend_flag = F.when(F.col("daily_spend_coins") > 0, 1).otherwise(0)
    df = df.withColumn("__has_spend__", spend_flag)
    df = df.withColumn("is_spender_7d", F.max("__has_spend__").over(w.rowsBetween(-7, -1)))

    # Days since last spend (999 if never)
    spend_date = F.when(F.col("daily_spend_coins") > 0, F.col("session_date"))
    last_spend = F.max(spend_date).over(w.rowsBetween(Window.unboundedPreceding, -1))
    df = df.withColumn("days_since_last_spend", F.when(last_spend.isNull(), 999).otherwise(F.datediff(F.col("session_date"), last_spend)))

    # Spending frequency in last 7 days
    df = df.withColumn("spending_frequency_7d", lagged_sum("__has_spend__", 7) / F.lit(7.0))

    # Avg spend per active day in last 7 days
    spend_nonzero = F.when(F.col("daily_spend_coins") > 0, F.col("daily_spend_coins"))
    avg_spend_active_7 = F.avg(spend_nonzero).over(w.rowsBetween(-7, -1))
    df = df.withColumn("avg_spend_per_active_day_7d", F.coalesce(avg_spend_active_7, F.lit(0.0)))

    # Spending volatility in last 7 days
    df = df.withColumn("spending_volatility_7d", F.coalesce(lagged_std("daily_spend_coins", 7), F.lit(0.0)))

    # 9) Win/loss streaks and win/loss ratio (use completion_rate proxy)
    win_flag = F.when(F.col("completion_rate") > 0.5, 1).otherwise(0)
    loss_flag = F.when(F.col("completion_rate") <= 0.5, 1).otherwise(0)
    df = df.withColumn("__win__", win_flag).withColumn("__loss__", loss_flag)

    # Consecutive wins streak (lagged)
    win_seg_start = F.when((F.col("__win__") == 1) & (F.lag("__win__", 1).over(w) != 1), 1).otherwise(0)
    win_seg_id = F.sum(win_seg_start).over(w.rowsBetween(Window.unboundedPreceding, 0))
    w_win_seg = Window.partitionBy("user_id", win_seg_id).orderBy("session_date")
    win_pos = F.when(F.col("__win__") == 1, F.row_number().over(w_win_seg)).otherwise(0)
    df = df.withColumn("__win_pos__", win_pos)
    df = df.withColumn("consecutive_wins_streak", F.coalesce(F.lag("__win_pos__", 1).over(w), F.lit(0)))

    # Consecutive losses streak (lagged)
    loss_seg_start = F.when((F.col("__loss__") == 1) & (F.lag("__loss__", 1).over(w) != 1), 1).otherwise(0)
    loss_seg_id = F.sum(loss_seg_start).over(w.rowsBetween(Window.unboundedPreceding, 0))
    w_loss_seg = Window.partitionBy("user_id", loss_seg_id).orderBy("session_date")
    loss_pos = F.when(F.col("__loss__") == 1, F.row_number().over(w_loss_seg)).otherwise(0)
    df = df.withColumn("__loss_pos__", loss_pos)
    df = df.withColumn("consecutive_losses_streak", F.coalesce(F.lag("__loss_pos__", 1).over(w), F.lit(0)))

    # Win/loss ratio (last 3 days, lagged)
    rolling_wins = lagged_sum("levels_completed", 3)
    rolling_attempts = lagged_sum("levels_started", 3)
    df = df.withColumn("win_loss_ratio_3d", F.when(rolling_attempts > 0, rolling_wins / rolling_attempts).otherwise(0.0))

    # Recent struggle indicator (yesterday completion_rate < 0.3)
    df = df.withColumn("recent_struggle_indicator", F.when(F.lag("completion_rate", 1).over(w) < 0.3, 1).otherwise(0))

    # 10) User lifecycle / cohort
    dsi = F.col("days_since_install")
    if "days_since_install" in df.columns:
        df = df.withColumn("lifecycle_new_user", F.when(dsi <= 7, 1).otherwise(0)) \
               .withColumn("lifecycle_early", F.when((dsi > 7) & (dsi <= 30), 1).otherwise(0)) \
               .withColumn("lifecycle_established", F.when((dsi > 30) & (dsi <= 90), 1).otherwise(0)) \
               .withColumn("lifecycle_veteran", F.when(dsi > 90, 1).otherwise(0))

    # Install month from first session date per user (use partition-only window)
    w_user_all = Window.partitionBy("user_id")
    df = df.withColumn("install_month", F.month(F.min("session_date").over(w_user_all)))

    # Whale detection using lagged cumulative spend (cumsum shifted by 1)
    cum_spend_lagged = F.sum(F.col("daily_spend_coins")).over(w.rowsBetween(Window.unboundedPreceding, -1))
    df = df.withColumn("total_lifetime_coins_spent", F.coalesce(cum_spend_lagged, F.lit(0.0)))
    # Global 90th percentile threshold
    whale_threshold = df.stat.approxQuantile("total_lifetime_coins_spent", [0.90], 0.01)[0]
    if whale_threshold is not None:
        df = df.withColumn("is_whale", F.when(F.col("total_lifetime_coins_spent") >= F.lit(whale_threshold), 1).otherwise(0))

    # 11) Difficulty trajectory
    # Slope over last 3 days (lagged): use OLS slope formula on row index
    t = F.row_number().over(w)
    df = df.withColumn("__t__", t)
    y_mult = F.col("current_effectivelevelmultiplier")
    x = F.col("__t__")
    xy = x * y_mult
    x2 = x * x
    sum_x = F.sum(x).over(w.rowsBetween(-3, -1))
    sum_y_mult = F.sum(y_mult).over(w.rowsBetween(-3, -1))
    sum_xy = F.sum(xy).over(w.rowsBetween(-3, -1))
    sum_x2 = F.sum(x2).over(w.rowsBetween(-3, -1))
    n3 = F.count(F.lit(1)).over(w.rowsBetween(-3, -1))
    denom = (n3 * sum_x2 - sum_x * sum_x)
    slope_mult = F.when(denom > 0, (n3 * sum_xy - sum_x * sum_y_mult) / denom).otherwise(0.0)
    df = df.withColumn("difficulty_trend_3d", F.coalesce(slope_mult, F.lit(0.0)))

    # Difficulty change magnitude (sum abs diffs over last 3 days, lagged)
    mult_diff = F.abs(y_mult - F.lag(y_mult, 1).over(w))
    df = df.withColumn("__mult_diff__", mult_diff)
    df = df.withColumn("difficulty_change_magnitude_3d", F.coalesce(F.sum("__mult_diff__").over(w.rowsBetween(-3, -1)), F.lit(0.0)))

    # Time at current difficulty (consecutive same multiplier; lagged)
    prev_mult = F.lag(y_mult, 1).over(w)
    diff_seg_start = F.when(prev_mult.isNull() | (F.abs(y_mult - prev_mult) >= 0.01), 1).otherwise(0)
    diff_seg_id = F.sum(diff_seg_start).over(w.rowsBetween(Window.unboundedPreceding, 0))
    w_diff_seg = Window.partitionBy("user_id", diff_seg_id).orderBy("session_date")
    time_at_curr = F.row_number().over(w_diff_seg)
    df = df.withColumn("__time_at_diff__", time_at_curr)
    df = df.withColumn("time_at_current_difficulty", F.coalesce(F.lag("__time_at_diff__", 1).over(w), F.lit(0)))

    # Distance from comfort zone (current vs preferred over 7d)
    df = df.withColumn("distance_from_comfort_zone", F.abs(F.col("current_effectivelevelmultiplier") - F.col("preferred_difficulty_7d")))

    # 12) Engagement patterns
    df = df.withColumn("session_intensity_3d", F.coalesce(lagged_avg("daily_event_count", 3), F.lit(0.0)))

    # Playtime trend (slope) over last 3 days (lagged)
    y_play = F.col("daily_play_duration_seconds")
    xy_p = x * y_play
    sum_y_p = F.sum(y_play).over(w.rowsBetween(-3, -1))
    sum_xy_p = F.sum(xy_p).over(w.rowsBetween(-3, -1))
    slope_play = F.when(denom > 0, (n3 * sum_xy_p - sum_x * sum_y_p) / denom).otherwise(0.0)
    df = df.withColumn("playtime_trend_3d", F.coalesce(slope_play, F.lit(0.0)))

    # Days since last play (gap length, lagged)
    gap_curr = F.when(prev_date.isNull(), 0).otherwise(F.datediff(F.col("session_date"), prev_date) - 1)
    df = df.withColumn("__gap_curr__", gap_curr)
    df = df.withColumn("days_since_last_play", F.coalesce(F.lag("__gap_curr__", 1).over(w), F.lit(0)))

    # Weekend warrior (avg weekend flag last 7 days, lagged)
    df = df.withColumn("weekend_warrior", F.when(lagged_avg("is_weekend", 7) > 0.5, 1).otherwise(0))

    # 13) Cross-feature interactions (current day context)
    df = df.withColumn("skill_x_difficulty", F.col("current_avg_attemptperuser") * F.col("current_effectivelevelmultiplier")) \
           .withColumn("spend_x_completion", F.col("daily_spend_coins") * F.col("completion_rate")) \
           .withColumn("playtime_x_skill", F.col("daily_play_duration_seconds") * F.col("current_avg_attemptperuser"))

    # Cleanup temp columns
    for c in [
        "__seg_id_days__", "__streak_days_curr__", "__high_engagement__",
        "__win__", "__loss__", "__win_pos__", "__loss_pos__",
        "__t__", "__mult_diff__", "__time_at_diff__", "__gap_curr__"
    ]:
        if c in df.columns:
            df = df.drop(c)

    # Fill NaNs in newly added advanced features to 0 (parity with pandas fillna(0))
    new_cols = [c for c in df.columns if c not in original_cols]
    numeric_types = {"int", "bigint", "double", "float", "decimal", "smallint", "tinyint"}
    new_numeric_cols = [c for c, t in df.dtypes if c in new_cols and any(nt in t for nt in numeric_types)]
    if new_numeric_cols:
        df = df.fillna(0, subset=new_numeric_cols)

    print("✅ Advanced gameplay features computed")
    return df


def write_partitioned_dataset(spark: SparkSession,
                              df: DataFrame,
                              out_dir: str,
                              partition_col: str,
                              fmt: str = "delta") -> None:
    """Write dataset in Delta or Parquet, partitioned by a single column (e.g., session_date)."""
    fmt = fmt.lower()
    if fmt not in {"delta", "parquet"}:
        raise ValueError("--output-format must be 'delta' or 'parquet'")

    print(f"💾 Writing {fmt.upper()} to {out_dir} partitioned by [{partition_col}]...")
    writer_base = (
        df.repartition(F.col(partition_col))
          .write
          .mode("overwrite")
          .option("compression", "snappy")
          .partitionBy(partition_col)
    )

    if fmt == "delta":
        try:
            writer_base.format("delta").save(out_dir)
            try:
                spark.sql(f"OPTIMIZE delta.`{out_dir}` ZORDER BY (user_id)")
                print("✅ OPTIMIZE + ZORDER executed")
            except Exception as opt_ex:
                print(f"⚠️  OPTIMIZE/ZORDER not executed: {opt_ex}")
                print("   Consider running OPTIMIZE using your Delta Lake environment post-write.")
            print("✅ Delta write complete")
        except Exception as e:
            msg = str(e)
            print(f"⚠️  Delta write failed: {msg}")
            print("   Falling back to Parquet write at the same output location...")
            writer_base.parquet(out_dir)
            print("✅ Parquet write complete (fallback)")
    else:
        writer_base.parquet(out_dir)
        print("✅ Parquet write complete")


def lowercase_all_columns(df: DataFrame) -> DataFrame:
    """Alias all columns to lowercase for consistent schema."""
    return df.select(*[F.col(c).alias(c.lower()) for c in df.columns])


def _spark_dtype_for(dt: DataType) -> DataType:
    """Utility to coerce nested numeric types to DoubleType for EWMA outputs."""
    if isinstance(dt, (IntegerType, LongType, FloatType, DecimalType)):
        return DoubleType()
    return dt


def compute_ewma_features_spark(df: DataFrame,
                                alpha_values: List[float] = [0.1, 0.3, 0.5, 0.7],
                                ewma_cols: List[str] = None) -> DataFrame:
    """
    Compute lagged EWMA features per user using a grouped map Pandas UDF (mapInPandas).
    - Strictly runs inside Spark executors via Arrow; no driver toPandas.
    - Parity with pandas: ewma(...).shift(1)
    - Excludes identifiers, lag features, and any 'next_*' columns.
    - If ewma_cols is provided, only those columns are used (reduces memory pressure).
    Returns a DataFrame with ['user_id','session_date', ewma_* columns].
    """
    print(f"📊 Computing EWMA features via grouped Pandas UDF for alphas={alpha_values}...")

    # Ensure we have normalized keys present
    assert 'user_id' in df.columns and 'session_date' in df.columns, "Expected user_id and session_date columns"

    # Determine numeric columns to include in EWMA
    exclude = {"user_id", "session_date", "difficulty_arm", "previous_day_action", "previous_day_multiplier"}
    numeric_types = (IntegerType, LongType, FloatType, DoubleType, DecimalType)
    
    if ewma_cols:
        # Use curated column list
        available_cols = set(df.columns)
        numeric_cols = [col for col in ewma_cols if col in available_cols and not col.startswith('next_') and col not in exclude]
        print(f"   Using curated EWMA columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}")
    else:
        # Auto-detect all numeric columns (original behavior)
        numeric_cols = [name for name, dtype in df.dtypes
                        if not name.startswith('next_') and name not in exclude and any(t in dtype for t in ['int', 'bigint', 'double', 'float', 'decimal', 'smallint', 'tinyint'])]
        print(f"   Auto-detected {len(numeric_cols)} numeric columns for EWMA")

    if not numeric_cols:
        print("   No numeric columns found for EWMA; skipping.")
        return df

    # Build schema for pandas UDF output
    # Keep user_id/session_date types from input; EWMA cols as DoubleType
    user_id_field = next((f for f in df.schema.fields if f.name == 'user_id'), None)
    session_date_field = next((f for f in df.schema.fields if f.name == 'session_date'), None)
    user_id_type = user_id_field.dataType if user_id_field else StringType()
    session_date_type = session_date_field.dataType if session_date_field else DateType()

    ewma_fields = []
    for col in numeric_cols:
        for a in alpha_values:
            ewma_fields.append(StructField(f"ewma_{col}_alpha{int(a*10)}", DoubleType(), True))

    ewma_schema = StructType([
        StructField('user_id', user_id_type, False),
        StructField('session_date', session_date_type, False),
        *ewma_fields
    ])

    def ewma_apply(pdf_iter):
        import pandas as pd
        import numpy as np
        for pdf in pdf_iter:
            # Sort to ensure stable EWMA by date
            if 'session_date' in pdf.columns:
                pdf = pdf.sort_values(['session_date'])
            out = pd.DataFrame({
                'user_id': pdf['user_id'].values,
                'session_date': pd.to_datetime(pdf['session_date']).values
            })
            for col in numeric_cols:
                series = pd.to_numeric(pdf[col], errors='coerce')
                for a in alpha_values:
                    out[f"ewma_{col}_alpha{int(a*10)}"] = (
                        series.ewm(alpha=a, adjust=False).mean().shift(1)
                    )
            # Fill NaNs with 0 for EWMA features
            for c in out.columns:
                if c not in ('user_id', 'session_date'):
                    out[c] = out[c].fillna(0.0)
            yield out

    # Group by user and compute EWMAs
    ewma_df = df.groupBy('user_id').applyInPandas(ewma_apply, schema=ewma_schema)

    # Join back with original df on keys
    joined = df.join(ewma_df, on=['user_id', 'session_date'], how='inner')
    print("✅ EWMA features computed and joined")
    return joined


def write_checkpoint_dataset(spark: SparkSession,
                            df: DataFrame,
                            checkpoint_dir: str,
                            name: str,
                            partition_col: str,
                            fmt: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_dir = os.path.join(checkpoint_dir, f"{name}.{fmt}")
    print(f"💾 Writing checkpoint [{name}] to {out_dir} ({fmt})...")
    write_partitioned_dataset(spark, df, out_dir, partition_col=partition_col, fmt=fmt)
    return out_dir


def read_checkpoint_dataset(spark: SparkSession,
                           checkpoint_dir: str,
                           name: str,
                           fmt: str):
    """Read a checkpoint with robust fallback across Delta/Parquet suffixes."""
    base = os.path.join(checkpoint_dir, name)
    # primary path with requested fmt
    primary_path = f"{base}.{fmt}"
    print(f"📥 Reading checkpoint [{name}] from {primary_path} ({fmt})...")
    try:
        reader_fmt = 'parquet' if fmt == 'parquet' else 'delta'
        return spark.read.format(reader_fmt).load(primary_path)
    except Exception as e:
        print(f"⚠️  Primary read failed: {e}")
        # try alternate suffix
        alt_fmt = 'parquet' if fmt == 'delta' else 'delta'
        alt_path = f"{base}.{alt_fmt}"
        print(f"🔁 Trying alternate checkpoint path: {alt_path} ({alt_fmt})")
        try:
            reader_fmt = 'parquet' if alt_fmt == 'parquet' else 'delta'
            return spark.read.format(reader_fmt).load(alt_path)
        except Exception as e2:
            # final attempt: read parquet at primary folder regardless of suffix
            try:
                print(f"🔁 Final attempt: reading as Parquet at {primary_path}")
                return spark.read.parquet(primary_path)
            except Exception:
                print(f"❌ Failed to read checkpoint [{name}] from {primary_path} or {alt_path}")
                raise e2


def main():
    parser = argparse.ArgumentParser(description="Daily Bandit Pipeline for SpacePlay (PURE SPARK)")
    parser.add_argument('--use-parquet', action='store_true', help='Load from Parquet instead of Snowflake')
    parser.add_argument('--use-csv', action='store_true', help='Load from CSV instead of Snowflake')
    parser.add_argument('--parquet-path', type=str, default='./data/spaceplay_raw.parquet', help='Path to Parquet export')
    parser.add_argument('--csv-path', type=str, default='./data/spaceplay_raw.csv', help='Path to CSV export directory')
    parser.add_argument('--out-dir', type=str, default='test_folder/daily_features_spark.delta', help='Output dataset directory (Delta by default)')
    parser.add_argument('--start-date', type=str, default='2025-07-01', help='Minimum gauserstartdate for Snowflake query (YYYY-MM-DD)')
    parser.add_argument('--output-format', type=str, default='delta', choices=['delta','parquet'], help='Output format for dataset')
    parser.add_argument('--partition-column', type=str, default='session_date', help='Column to partition the output by')
    parser.add_argument('--shuffle-partitions', type=int, default=2000, help='spark.sql.shuffle.partitions value')
    parser.add_argument('--local-dirs', type=str, default='', help='Comma-separated local dirs for shuffle spill (spark.local.dir). WARNING: /tmp may be small/in-memory; use large disk partitions for big runs.')
    parser.add_argument('--disable-aqe', action='store_true', help='Disable Adaptive Query Execution (AQE)')
    # EWMA options for scale
    parser.add_argument('--disable-ewma', action='store_true', help='Skip EWMA feature computation (saves memory/time on large runs)')
    parser.add_argument('--ewma-cols', type=str, default='', help='Comma-separated list of columns to compute EWMA on (if not provided, all numeric columns are used). Ignored if --disable-ewma is set.')
    # Checkpointing options
    parser.add_argument('--enable-checkpoints', action='store_true', help='Enable writing intermediate checkpoints (daily_features, daily_bandit, advanced_features)')
    parser.add_argument('--checkpoint-dir', type=str, default='test_folder/checkpoints', help='Directory to store checkpoints')
    parser.add_argument('--resume-from', type=str, default='none', choices=['none','daily_features','daily_bandit','advanced_features'], help='Resume from an intermediate checkpoint')
    # Snowflake connector options
    parser.add_argument('--use-sf-connector', action='store_true', help='Use Spark Snowflake Connector instead of JDBC')
    parser.add_argument('--sf-url', type=str, default='', help='Snowflake URL, e.g., https://<account>.snowflakecomputing.com (overrides JSON)')
    parser.add_argument('--sf-user', type=str, default='', help='Snowflake user (overrides JSON)')
    parser.add_argument('--sf-password', type=str, default='', help='Snowflake password (overrides JSON)')
    parser.add_argument('--sf-account', type=str, default='', help='Snowflake account (optional if derivable; overrides JSON)')
    parser.add_argument('--sf-warehouse', type=str, default='DASHBOARD_DEFAULT', help='Snowflake warehouse')
    parser.add_argument('--sf-database', type=str, default='SPACEPLAY', help='Snowflake database')
    parser.add_argument('--sf-schema', type=str, default='UNITY', help='Snowflake schema')
    parser.add_argument('--sf-role', type=str, default='', help='Snowflake role (optional)')
    parser.add_argument('--sf-packages', type=str, default='net.snowflake:spark-snowflake_2.13:3.1.4,net.snowflake:snowflake-jdbc:3.16.1', help='Maven coordinates for Snowflake connector packages (match Spark/Scala)')
    parser.add_argument('--delta-packages', type=str, default='io.delta:delta-spark_2.13:4.0.0', help='Maven coordinates for Delta Lake (match Spark/Scala)')
    parser.add_argument('--sf-config-json', type=str, default='config/snowflake.json', help='Path to Snowflake JSON config (auto-loaded if present)')
    args = parser.parse_args()

    print("🚀 Starting Daily Bandit Pipeline (PURE SPARK)")
    print("=" * 80)
    if args.use_parquet:
        print(f"📂 DATA SOURCE: Parquet ({args.parquet_path})")
    elif args.use_csv:
        print(f"📂 DATA SOURCE: CSV ({args.csv_path})")
    else:
        print(f"📂 DATA SOURCE: Snowflake (gauserstartdate ≥ {args.start_date}) via {'Connector' if args.use_sf_connector else 'JDBC'}")
    print("=" * 80)

    # Validate and warn about local-dirs (shuffle spill)
    if args.local_dirs:
        local_dir_list = [d.strip() for d in args.local_dirs.split(',') if d.strip()]
        if '/tmp' in local_dir_list and len(local_dir_list) == 1:
            print("⚠️  WARNING: Using /tmp for shuffle spill. For large runs (4+ months), this may fill up.")
            print("   Consider using larger disk partitions: --local-dirs '/mnt/ssd1/spark,/mnt/ssd2/spark'")
            print("   Alternative: Use main disk: --local-dirs '$HOME/spark-temp' (create directory first)")
        # Check disk space for each directory and verify they exist
        for local_dir in local_dir_list:
            try:
                import os
                if not os.path.exists(local_dir):
                    print(f"❌ ERROR: Local directory does not exist: {local_dir}")
                    print(f"   Please create it first: mkdir -p {local_dir}")
                    raise ValueError(f"Local directory does not exist: {local_dir}")
                statvfs = os.statvfs(local_dir)
                free_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
                if free_gb < 100:
                    print(f"⚠️  WARNING: {local_dir} has only {free_gb:.1f}GB free. May be insufficient for large shuffles.")
                    print(f"   Recommended: At least 200GB free for 2+ months of data")
            except ValueError:
                raise
            except Exception as e:
                print(f"⚠️  Could not check disk space for {local_dir}: {e}")
    else:
        print("⚠️  WARNING: No --local-dirs specified. Spark will use default temp directory (may be /tmp).")
        print("   For large runs, specify: --local-dirs '/path/to/large/disk/spark'")
    
    # Resume summary (UX)
    if args.resume_from != 'none':
        print("\n" + "=" * 80)
        print("🔄 RESUMING PIPELINE")
        print("=" * 80)
        print(f"   Checkpoint: {args.resume_from}")
        print(f"   Location:   {args.checkpoint_dir}")
        if args.resume_from == 'daily_bandit':
            print("   Skipping: Data loading, daily aggregation, logged actions, normalization/cleaning")
            print("   Running:  Advanced features → EWMA → Write")
        elif args.resume_from == 'daily_features':
            print("   Skipping: Data loading, daily aggregation")
            print("   Running:  Logged actions → Daily bandit → Normalization → Advanced → EWMA → Write")
        elif args.resume_from == 'advanced_features':
            print("   Skipping: Data loading, daily aggregation, logged actions, normalization, advanced features")
            print("   Running:  EWMA → Write")
        print("=" * 80 + "\n")

    # Combine packages: Delta Lake (if delta output) + Snowflake connector (if used)
    packages = []
    if args.output_format == 'delta':
        packages.append(args.delta_packages)
    if args.use_sf_connector and args.sf_packages:
        packages.append(args.sf_packages)
    combined_packages = ",".join(packages)
    
    spark = initialize_spark_session(
        shuffle_partitions=args.shuffle_partitions,
        local_dirs=args.local_dirs,
        enable_aqe=not args.disable_aqe,
        extra_packages=combined_packages,
        enable_delta_extensions=(args.output_format == 'delta')
    )

    try:
        # 1) Load/resume
        resume = args.resume_from
        checkpoint_dir = args.checkpoint_dir

        if resume == 'daily_bandit':
            print("🔁 Resuming from checkpoint: daily_bandit")
            daily_bandit = read_checkpoint_dataset(spark, checkpoint_dir, 'daily_bandit', 'delta' if args.output_format=='delta' else 'parquet')
        else:
            if args.use_parquet:
                raw_events = load_from_parquet(spark, args.parquet_path)
            elif args.use_csv:
                raw_events = load_from_csv(spark, args.csv_path)
            else:
                # Validate start-date format (YYYY-MM-DD)
                try:
                    datetime.strptime(args.start_date, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"Invalid --start-date '{args.start_date}'. Expected format YYYY-MM-DD.")

                # Try loading JSON config if present
                json_cfg = load_snowflake_config_from_json(args.sf_config_json)

                if args.use_sf_connector:
                    # Build connector options with priority: CLI > JSON > in-script defaults
                    sf_url = args.sf_url or (json_cfg.get('url') if json_cfg and json_cfg.get('url') else '')
                    sf_user = args.sf_user or (json_cfg.get('user') if json_cfg and json_cfg.get('user') else '')
                    sf_password = args.sf_password or (json_cfg.get('password') if json_cfg and json_cfg.get('password') else '')
                    sf_account = args.sf_account or (json_cfg.get('account') if json_cfg and json_cfg.get('account') else '')
                    sf_warehouse = args.sf_warehouse or (json_cfg.get('warehouse') if json_cfg and json_cfg.get('warehouse') else 'DASHBOARD_DEFAULT')
                    sf_database = args.sf_database or (json_cfg.get('database') if json_cfg and json_cfg.get('database') else 'SPACEPLAY')
                    sf_schema = args.sf_schema or (json_cfg.get('schema') if json_cfg and json_cfg.get('schema') else 'UNITY')
                    sf_role = args.sf_role or (json_cfg.get('role') if json_cfg and json_cfg.get('role') else '')

                    # Fallback to hardcoded JDBC defaults if still missing
                    if not sf_user or not sf_password or (not sf_url and not sf_account):
                        _cfg, _ = connect_to_snowflake_direct()
                        sf_user = sf_user or _cfg.get('user')
                        sf_password = sf_password or _cfg.get('password')
                        sf_account = sf_account or _cfg.get('account')
                        sf_url = sf_url or (f"https://{sf_account}.snowflakecomputing.com" if sf_account else '')
                        sf_warehouse = sf_warehouse or _cfg.get('warehouse')
                        sf_database = sf_database or _cfg.get('database')
                        sf_schema = sf_schema or _cfg.get('schema')

                    if not sf_url or not sf_user or not sf_password:
                        raise ValueError("Missing Snowflake connector credentials. Provide CLI/JSON or rely on hardcoded defaults.")

                    sf_options = {
                        "sfURL": sf_url,
                        "sfUser": sf_user,
                        "sfPassword": sf_password,
                        "sfWarehouse": sf_warehouse,
                        "sfDatabase": sf_database,
                        "sfSchema": sf_schema,
                    }
                    if sf_role:
                        sf_options["sfRole"] = sf_role
                    if sf_account:
                        sf_options["sfAccount"] = sf_account
                    raw_events = create_snowflake_connector_dataframe(spark, sf_options, start_date=args.start_date)
                else:
                    if json_cfg:
                        snowflake_config = {
                            "user": json_cfg.get('user') or "",
                            "password": json_cfg.get('password') or "",
                            "account": json_cfg.get('account') or (json_cfg.get('url').split('https://')[-1].split('.snowflakecomputing.com')[0] if json_cfg.get('url') else ""),
                            "warehouse": json_cfg.get('warehouse') or 'DASHBOARD_DEFAULT',
                            "database": json_cfg.get('database') or 'SPACEPLAY',
                            "schema": json_cfg.get('schema') or 'UNITY',
                        }
                        if not snowflake_config['user'] or not snowflake_config['password'] or not snowflake_config['account']:
                            print("⚠️  Incomplete JDBC config from JSON; falling back to in-script defaults")
                            snowflake_config, jdbc_url = connect_to_snowflake_direct()
                        else:
                            jdbc_url = f"jdbc:snowflake://{snowflake_config['account']}.snowflakecomputing.com/"
                    else:
                        snowflake_config, jdbc_url = connect_to_snowflake_direct()
                    raw_events = create_spark_dataframe(spark, snowflake_config, jdbc_url, start_date=args.start_date)

        if resume == 'daily_bandit':
            pass
        else:
            # 2) Prepare dates and filter to multi-day users
            events_with_date = add_session_date_column(raw_events)
            multi_day_events = retain_multi_day_users(events_with_date)

            # 3) Logged actions and arms
            logged_actions = extract_logged_action(multi_day_events)
            arm_assignments = assign_difficulty_arms(logged_actions)

            if resume == 'daily_features':
                print("🔁 Resuming from daily_features checkpoint")
                daily_features = read_checkpoint_dataset(spark, checkpoint_dir, 'daily_features', 'delta' if args.output_format=='delta' else 'parquet')
            else:
                # 4) Aggregate core daily features
                daily_features = aggregate_daily_features(multi_day_events)
                if args.enable_checkpoints:
                    write_checkpoint_dataset(spark, daily_features, checkpoint_dir, 'daily_features', partition_col=args.partition_column, fmt=args.output_format)

            # 5) Create bandit dataset (features + action + reward)
            daily_bandit = create_daily_bandit_dataset(
                daily_features,
                logged_actions,
                arm_assignments
            )

        # 6) Normalize columns, clean low-variance columns (Spark-only)
        if resume != 'daily_bandit':
            daily_bandit = normalize_to_user_lowercase(daily_bandit)
            daily_bandit = clean_null_and_zero_columns_spark(daily_bandit, id_cols=["user_id", "session_date", "difficulty_arm"])
            # FORCE MATERIALIZATION: Write and immediately read back to break Snowflake connector lineage
            # This prevents the connector from being accessed during advanced features computation
            if args.enable_checkpoints:
                write_checkpoint_dataset(spark, daily_bandit, checkpoint_dir, 'daily_bandit', partition_col=args.partition_column, fmt=args.output_format)
                print("💾 Materializing daily_bandit to break Spark lineage from Snowflake connector...")
                daily_bandit = read_checkpoint_dataset(spark, checkpoint_dir, 'daily_bandit', 'delta' if args.output_format=='delta' else 'parquet')
            elif args.use_sf_connector and not args.use_parquet and not args.use_csv:
                # Even without explicit checkpoints, force materialization when using Snowflake connector
                # to avoid telemetry issues during advanced features
                print("💾 Auto-materializing daily_bandit to break Snowflake connector lineage...")
                temp_checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else 'test_folder/checkpoints'
                os.makedirs(temp_checkpoint_dir, exist_ok=True)
                write_checkpoint_dataset(spark, daily_bandit, temp_checkpoint_dir, 'daily_bandit', partition_col=args.partition_column, fmt=args.output_format)
                daily_bandit = read_checkpoint_dataset(spark, temp_checkpoint_dir, 'daily_bandit', 'delta' if args.output_format=='delta' else 'parquet')
        elif args.enable_checkpoints:
            # Resuming: already materialized
            pass

        # 7) Advanced gameplay features (Spark windows)
        if resume == 'advanced_features':
            print("🔁 Resuming from checkpoint: advanced_features")
            daily_with_adv = read_checkpoint_dataset(spark, checkpoint_dir, 'advanced_features', 'delta' if args.output_format=='delta' else 'parquet')
        else:
            daily_with_adv = add_advanced_gameplay_features_spark(daily_bandit)
            if args.enable_checkpoints and resume not in ('daily_bandit','advanced_features'):
                write_checkpoint_dataset(spark, daily_with_adv, checkpoint_dir, 'advanced_features', partition_col=args.partition_column, fmt=args.output_format)

        # 8) EWMA features via grouped Pandas UDF (fast and vectorized inside Spark)
        if args.disable_ewma:
            print("⏭️  Skipping EWMA features (--disable-ewma set)")
        else:
            ewma_cols_list = [c.strip() for c in args.ewma_cols.split(',')] if args.ewma_cols else None
            daily_with_adv = compute_ewma_features_spark(daily_with_adv, ewma_cols=ewma_cols_list)

        # 9) Lowercase all columns for consistency
        daily_with_adv = lowercase_all_columns(daily_with_adv)

        # 10) Write dataset partitioned by date only (recommended to reduce small files)
        write_partitioned_dataset(
            spark,
            daily_with_adv,
            args.out_dir,
            partition_col=args.partition_column,
            fmt=args.output_format,
        )

        print("\n" + "=" * 80)
        print("📊 FINAL SUMMARY (PURE SPARK)")
        print("=" * 80)
        print(f"   Output directory: {args.out_dir}")
        print(f"   Partition column: {args.partition_column}")
        print(f"   Output format: {args.output_format}")
        print("✅ Daily bandit pipeline (pure Spark) complete!")

    except Exception as exc:
        print(f"❌ Pipeline failed: {exc}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        spark.stop()
        print("🛑 Spark session stopped")


if __name__ == "__main__":
    main()
