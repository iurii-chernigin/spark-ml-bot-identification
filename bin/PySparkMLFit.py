import operator
import argparse

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier

MODEL_PATH = 'spark_ml_model'
LABEL_COL = 'is_bot'

def build_grid_params(estimator) -> ParamGridBuilder:
    return ParamGridBuilder() \
                .addGrid(estimator.maxDepth, [2, 3, 4]) \
                .addGrid(estimator.maxBins, [3, 4]) \
                .build()

def build_cross_validator(pipeline, grid_params, evaluator) -> CrossValidator:
    return CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=grid_params,
        evaluator=evaluator,
        parallelism=2
    ) 

def build_evaluator() -> MulticlassClassificationEvaluator:
    return MulticlassClassificationEvaluator(
        predictionCol='prediction',
        labelCol='is_bot',
        metricName='accuracy'
    )

def vector_assembler(features) -> VectorAssembler:
    return VectorAssembler(inputCols=features, outputCol='features')


def process(spark, data_path, model_path):
    """
    Основной процесс задачи.

    :param spark: SparkSession
    :param data_path: путь до датасета
    :param model_path: путь сохранения обученной модели
    """
    # Estimators to prepare the data
    user_type_index = StringIndexer(inputCol='user_type', outputCol='user_type_index') 
    platform_index = StringIndexer(inputCol='platform', outputCol='platform_index')
    features_vector = vector_assembler([
        'user_type_index', 'platform_index', 'duration', 'events_per_min',
        'item_info_events', 'select_item_events', 'make_order_events' 
    ])
    # Model estimator
    dtc_estimator = DecisionTreeClassifier(labelCol='is_bot', featuresCol='features')
    pipeline = Pipeline(stages=[user_type_index, platform_index, features_vector, dtc_estimator])
    # Cros Validator
    cv = build_cross_validator(pipeline, build_grid_params(dtc_estimator), build_evaluator())
    # Train and save model
    train_df = spark.read.parquet(data_path)
    cv_model = cv.fit(train_df)
    cv_model.write().save(model_path)
    

def main(data_path, model_path):
    spark = _spark_session()
    process(spark, data_path, model_path)


def _spark_session():
    """
    Создание SparkSession.

    :return: SparkSession
    """
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='session-stat.parquet', help='Please set datasets path.')
    parser.add_argument('--model_path', type=str, default=MODEL_PATH, help='Please set model path.')
    args = parser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    main(data_path, model_path)
