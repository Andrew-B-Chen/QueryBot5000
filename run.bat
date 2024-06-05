REM Copy and decompress the sample data file
tar -xvzf "tiramisu-sample.tar.gz"

REM Generate and combine query templates
pre-processor\templatizer.py tiramisu --dir tiramisu-sample --output templates
pre-processor\csv-combiner.py --input_dir templates --output_dir tiramisu-combined-csv

REM Run through clustering algorithm
clusterer\online_clustering.py --dir tiramisu-combined-csv --rho 0.8
clusterer\generate-cluster-coverage.py --project tiramisu --assignment online-clustering-results\None-0.8-assignments.pickle --output_csv_dir online-clusters --output_dir cluster-coverage\

REM Run forecasting models (6-hour trace)
forecaster\exp_multi_online_continuous.py tiramisu --method ar --aggregate 10 --horizon 360 --input_dir online-clusters --cluster_path cluster-coverage\coverage.pickle --output_dir prediction-results
forecaster\exp_multi_online_continuous.py tiramisu --method kr --aggregate 10 --horizon 360 --input_dir online-clusters --cluster_path cluster-coverage\coverage.pickle --output_dir prediction-results
forecaster\exp_multi_online_continuous.py tiramisu --method rnn --aggregate 10 --horizon 360 --input_dir online-clusters --cluster_path cluster-coverage\coverage.pickle --output_dir prediction-results
forecaster\exp_multi_online_continuous.py tiramisu --method brr --aggregate 10 --horizon 360 --input_dir online-clusters --cluster_path cluster-coverage\coverage.pickle --output_dir prediction-results
forecaster\exp_multi_online_continuous.py tiramisu --method svr --aggregate 10 --horizon 360 --input_dir online-clusters --cluster_path cluster-coverage\coverage.pickle --output_dir prediction-results

REM Generate ENSEMBLE and HYBRID results (6-hour trace)
forecaster\generate_ensemble_hybrid.py prediction-results\agg-10\horizon-360\ar prediction-results\agg-10\horizon-360\noencoder-rnn prediction-results\agg-10\horizon-360\ensemble False
forecaster\generate_ensemble_hybrid.py prediction-results\agg-10\horizon-360\ensemble prediction-results\agg-10\horizon-360\kr prediction-results\agg-10\horizon-360\hybrid True

REM Plot results
forecaster\plot-prediction-accuracy.py prediction-results False
