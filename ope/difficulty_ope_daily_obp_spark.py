#!/usr/bin/env python3
"""
OPE on Spark Delta (No pandas)
==============================

Spark-native OPE that loads a single combined dataset (core + advanced + EWMA)
from Delta/Parquet and computes IPW, SNIPS, and DR estimates entirely with
Spark ML + DataFrame ops (no toPandas conversion).

Assumptions:
- Dataset contains: user_id, session_date, action (int 0..4), next_day_reward,
  difficulty_arm (string), current_effectivelevelmultiplier, and feature columns
- We evaluate a uniform target policy over valid arms per row (bounds [0.5,1.25])
- Propensity model: RandomForestClassifier (Spark ML)
- Reward models: one RandomForestRegressor per action
- Group K-fold split by user via hash-based fold assignment

Outputs aggregated metrics per fold and overall averages; no bootstrapping.
"""

import os
import sys
import argparse
import warnings
import json
from typing import Dict, List, Any

warnings.filterwarnings("ignore")

# Reuse the core OPE components by importing from the compatible script
ARM_ORDER = ["Easierer", "Easier", "Same", "Harder", "Harderer"]


def load_spark_dataset(delta_path: str,
                       use_parquet: bool = False,
                       spark_packages: str = "io.delta:delta-spark_2.12:3.1.0",
                       min_date: str = None):
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    print("⚡ Initializing Spark (reader only)...")
    builder = (SparkSession.builder
               .appName("OPE_Spark")
               .config("spark.sql.session.timeZone", "UTC"))
    if not use_parquet and spark_packages:
        builder = (builder
                   .config("spark.jars.packages", spark_packages)
                   .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
                   .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog"))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    fmt = "parquet" if use_parquet else "delta"
    print(f"📂 Reading {fmt.upper()} dataset: {delta_path}")
    try:
        df = spark.read.format(fmt).load(delta_path)
    except Exception as e:
        if not use_parquet:
            print(f"⚠️ Delta read failed ({e}); retrying as Parquet...")
            df = spark.read.parquet(delta_path)
        else:
            raise
    if min_date:
        print(f"📅 Filtering session_date >= {min_date}")
        df = df.filter(F.col('session_date') >= F.lit(min_date))
    return spark, df


def make_uniform_target_pi_df(df, lo=0.5, hi=1.25, n_arms=5):
    from pyspark.sql import functions as F, types as T
    def compute_pi(mult):
        # ARM_ORDER deltas
        deltas = [-0.12, -0.06, 0.0, 0.06, 0.12]
        valid = 0
        flags = [0]*n_arms
        if mult is None:
            return [1.0/n_arms]*n_arms
        for i, d in enumerate(deltas):
            nm = float(mult) + d
            if lo <= nm <= hi:
                flags[i] = 1
                valid += 1
        if valid == 0:
            return [1.0/n_arms]*n_arms
        return [f/valid for f in flags]
    udf_pi = F.udf(compute_pi, T.ArrayType(T.DoubleType(), containsNull=False))
    return df.withColumn('pi', udf_pi(F.col('current_effectivelevelmultiplier')))


def train_propensity(train_df, val_df, feature_cols: List[str]):
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.classification import RandomForestClassifier
    from pyspark.sql import functions as F, types as T
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    tr = assembler.transform(train_df).withColumn('label', F.col('action').cast('double'))
    va = assembler.transform(val_df)
    clf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=100, maxDepth=10, seed=42)
    model = clf.fit(tr)
    pred = model.transform(va)  # has probability: Vector
    # pscore = probability[action]
    def pick_prob(prob, idx):
        try:
            return float(prob[idx])
        except Exception:
            return 1e-6
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
    pick_udf = F.udf(pick_prob, T.DoubleType())
    pred = pred.withColumn('pscore', F.greatest(F.lit(1e-6), pick_udf(F.col('probability'), F.col('action'))))
    return pred


def train_reward_models(train_df, val_df, feature_cols: List[str], n_arms: int = 5):
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.regression import RandomForestRegressor
    from pyspark.sql import functions as F
    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features')
    tr = assembler.transform(train_df)
    va = assembler.transform(val_df)
    out = va
    for j in range(n_arms):
        tr_j = tr.filter(F.col('action') == j)
        count_j = tr_j.count()
        if count_j < 10:
            # too small, fallback to global mean
            mean_val = tr.select(F.mean('next_day_reward')).first()[0] or 0.0
            out = out.withColumn(f'q_hat_{j}', F.lit(float(mean_val)))
            continue
        rfr = RandomForestRegressor(labelCol='next_day_reward', featuresCol='features', numTrees=100, maxDepth=10, seed=42)
        model = rfr.fit(tr_j)
        out = model.transform(out).withColumnRenamed('prediction', f'q_hat_{j}')
    return out


def compute_estimators(df):
    from pyspark.sql import functions as F
    # pi_obs = element of pi at index action
    def pick_arr(arr, idx):
        try:
            return float(arr[idx])
        except Exception:
            return 0.0
    pick_arr_udf = F.udf(pick_arr, 'double')
    df = df.withColumn('pi_obs', pick_arr_udf(F.col('pi'), F.col('action')))
    df = df.withColumn('w', F.col('pi_obs') / F.col('pscore'))
    # q_hat_sum_pi = sum_j pi[j]*q_hat_j
    exprs = [F.col('pi')[i] * F.col(f'q_hat_{i}') for i in range(5) if f'q_hat_{i}' in df.columns]
    df = df.withColumn('q_hat_sum_pi', sum(exprs))
    # q_hat_a_obs via pick
    def pick_q(dfrow):
        pass
    def pick_qhat(cols, idx):
        try:
            return float(cols[int(idx)])
        except Exception:
            return 0.0
    from pyspark.sql import types as T
    # Create array of qhat columns
    qhat_cols = [F.col(f'q_hat_{i}') for i in range(5) if f'q_hat_{i}' in df.columns]
    df = df.withColumn('qhat_arr', F.array(*qhat_cols))
    pick_q_udf = F.udf(pick_qhat, T.DoubleType())
    df = df.withColumn('q_hat_obs', pick_q_udf(F.col('qhat_arr'), F.col('action')))

    # Metrics
    # IPW: mean(w * r)
    ipw = df.select(F.mean(F.col('w') * F.col('next_day_reward')).alias('ipw')).first()['ipw']
    # SNIPS: sum(w*r)/sum(w)
    agg = df.select(F.sum(F.col('w') * F.col('next_day_reward')).alias('num'), F.sum('w').alias('den')).first()
    snips = float(agg['num'])/float(agg['den']) if agg['den'] and agg['den'] != 0 else None
    # DR: mean(q_hat_sum_pi + (r - q_hat_obs) * w)
    dr = df.select(F.mean(F.col('q_hat_sum_pi') + (F.col('next_day_reward') - F.col('q_hat_obs')) * F.col('w')).alias('dr')).first()['dr']
    # Baseline: mean(r)
    baseline = df.select(F.mean('next_day_reward').alias('base')).first()['base']
    return {
        'ipw': float(ipw) if ipw is not None else None,
        'snips': float(snips) if snips is not None else None,
        'dr': float(dr) if dr is not None else None,
        'baseline': float(baseline) if baseline is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(description="OPE on Spark dataset (Delta/Parquet, Spark-native)")
    parser.add_argument('--delta-path', type=str, default='test_folder/daily_features_spark.delta', help='Path to Delta (or Parquet) dataset')
    parser.add_argument('--use-parquet', action='store_true', help='Force read as Parquet instead of Delta')
    parser.add_argument('--delta-packages', type=str, default='io.delta:delta-spark_2.12:3.1.0', help='Delta Lake package for Spark reader')
    parser.add_argument('--min-date', type=str, default=None, help='Filter to session_date >= min-date (YYYY-MM-DD)')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--output-dir', type=str, default='test_folder')
    args = parser.parse_args()

    # 1) Load dataset as Spark DataFrame
    spark, sdf = load_spark_dataset(
        delta_path=args.delta_path,
        use_parquet=args.use_parquet,
        spark_packages=args.delta_packages,
        min_date=args.min_date,
    )

    from pyspark.sql import functions as F

    # Ensure required columns
    required = ['user_id','session_date','action','next_day_reward','current_effectivelevelmultiplier']
    missing = [c for c in required if c not in sdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # If action is not integer, map from difficulty_arm
    if dict(sdf.dtypes).get('action') not in ('int', 'bigint'):  # rough check
        if 'difficulty_arm' in sdf.columns:
            print("🔢 Mapping difficulty_arm to integer action indices...")
            mapping = {"Easierer":0, "Easier":1, "Same":2, "Harder":3, "Harderer":4}
            mapping_expr = F.create_map([F.lit(x) for kv in mapping.items() for x in kv])
            sdf = sdf.withColumn('action', mapping_expr[F.col('difficulty_arm')].cast('int'))
        else:
            raise ValueError("Neither integer 'action' nor 'difficulty_arm' present to derive action indices.")

    # Select features: numeric columns excluding reserved
    numeric_types = {'int','bigint','double','float','decimal','smallint','tinyint'}
    reserved = {'user_id','session_date','difficulty_arm','next_day_reward','current_effectivelevelmultiplier','day_start_ts','day_end_ts','action'}
    feature_cols = [c for c,t in sdf.dtypes if (any(nt in t for nt in numeric_types) and c not in reserved)]
    print(f"📊 Using {len(feature_cols)} features")

    # Assign folds by user hash
    folds = max(2, int(args.folds))
    sdf = sdf.withColumn('fold_id', F.pmod(F.abs(F.hash(F.col('user_id'))), F.lit(folds)))

    fold_metrics: List[Dict[str, Any]] = []

    for k in range(folds):
        print(f"\n📦 Fold {k+1}/{folds}")
        tr = sdf.filter(F.col('fold_id') != F.lit(k))
        va = sdf.filter(F.col('fold_id') == F.lit(k))

        # Target policy pi (uniform over valid arms)
        va_pi = make_uniform_target_pi_df(va)

        # Propensity model
        va_ps = train_propensity(tr, va_pi, feature_cols)

        # Reward models (per action)
        va_q = train_reward_models(tr, va_ps, feature_cols, n_arms=5)

        # Estimator metrics
        metrics = compute_estimators(va_q)
        metrics['fold'] = k+1
        fold_metrics.append(metrics)
        print(f"   Fold metrics: {metrics}")

    # Aggregate
    def avg(name):
        vals = [m[name] for m in fold_metrics if m.get(name) is not None]
        return float(sum(vals)/len(vals)) if vals else None
    aggregated = {
        'ipw': avg('ipw'),
        'snips': avg('snips'),
        'dr': avg('dr'),
        'baseline': avg('baseline'),
        'n_folds': folds,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, 'spark_ope_results.json')
    with open(out_path, 'w') as f:
        json.dump({'folds': fold_metrics, 'aggregated': aggregated}, f, indent=2)
    print(f"✅ OPE complete. Saved: {out_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
