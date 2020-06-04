package com.yeahmobi
// Spark Training Pipeline Libraries
import org.apache.spark.ml.feature._
import org.apache.spark.ml.classification.{LogisticRegression, RandomForestClassifier}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.sql.SparkSession
import com.databricks.spark.avro._

// MLeap/Bundle.ML Serialization Libraries
import ml.combust.mleap.spark.SparkSupport._
import resource._
import ml.combust.bundle.BundleFile
import org.apache.spark.ml.bundle.SparkBundleContext

object test {
  def main(args: Array[String]): Unit = {
    val inputFile = "s3://algo.yeahmobi.com/etl/test/lending_club.avro"
    val spark = SparkSession
      .builder()
      .appName("mleapDemo")
      .getOrCreate()

    //Step 1 load data and preprocess
    var dataset = spark.sqlContext.read.format("com.databricks.spark.avro").
      load(inputFile)

    dataset.createOrReplaceTempView("df")
    println(dataset.count())

    val datasetFnl = spark.sqlContext.sql(f"""
    select
        loan_amount,
        fico_score_group_fnl,
        case when dti >= 10.0
            then 10.0
            else dti
        end as dti,
        emp_length,
        case when state in ('CA', 'NY', 'MN', 'IL', 'FL', 'WA', 'MA', 'TX', 'GA', 'OH', 'NJ', 'VA', 'MI')
            then state
            else 'Other'
        end as state,
        loan_title,
        approved
    from df
    where loan_title in('Debt Consolidation', 'Other', 'Home/Home Improvement', 'Payoff Credit Card', 'Car Payment/Loan',
    'Business Loan', 'Health/Medical', 'Moving', 'Wedding/Engagement', 'Vacation', 'College', 'Renewable Energy', 'Payoff Bills',
    'Personal Loan', 'Motorcycle')
""")
    println(datasetFnl.count())

    // Step 2: Define continous and categorical features and filter nulls
    val continuousFeatures = Array("loan_amount",
      "dti")

    val categoricalFeatures = Array("loan_title",
      "emp_length",
      "state",
      "fico_score_group_fnl")

    val allFeatures = continuousFeatures.union(categoricalFeatures)

    // Filter all null values
    val allCols = allFeatures.union(Seq("approved")).map(datasetFnl.col)
    val nullFilter = allCols.map(_.isNotNull).reduce(_ && _)
    val datasetImputedFiltered = datasetFnl.select(allCols: _*).filter(nullFilter).persist()

    println(datasetImputedFiltered.count())

    //Step 3: Split data into training and validation¶
    val Array(trainingDataset, validationDataset) = datasetImputedFiltered.randomSplit(Array(0.7, 0.3))

    //Step 4: Continous Feature Pipeline
    val continuousFeatureAssembler = new VectorAssembler(uid = "continuous_feature_assembler").
      setInputCols(continuousFeatures).
      setOutputCol("unscaled_continuous_features")

    val continuousFeatureScaler = new StandardScaler(uid = "continuous_feature_scaler").
      setInputCol("unscaled_continuous_features").
      setOutputCol("scaled_continuous_features")

    val polyExpansionAssembler = new VectorAssembler(uid = "poly_expansion_feature_assembler").
      setInputCols(Array("loan_amount", "dti")).
      setOutputCol("poly_expansions_features")

    val continuousFeaturePolynomialExpansion = new PolynomialExpansion(uid = "polynomial_expansion_loan_amount").
      setInputCol("poly_expansions_features").
      setOutputCol("loan_amount_polynomial_expansion_features")

    //Step 5: Categorical Feature Pipeline
    val categoricalFeatureIndexers = categoricalFeatures.map {
      feature => new StringIndexer(uid = s"string_indexer_$feature").
        setInputCol(feature).
        setOutputCol(s"${feature}_index")
    }

    val categoricalFeatureOneHotEncoders = categoricalFeatureIndexers.map {
      indexer => new OneHotEncoder(uid = s"oh_encoder_${indexer.getOutputCol}").
        setInputCol(indexer.getOutputCol).
        setOutputCol(s"${indexer.getOutputCol}_oh")
    }

    //Step 6: Assemble our features and feature pipeline
    val featureColsLr = categoricalFeatureOneHotEncoders.map(_.getOutputCol).union(Seq("scaled_continuous_features"))

    //Step 7: assemble all processes categorical and continuous features into a single feature vector
    val featureAssemblerLr = new VectorAssembler(uid = "feature_assembler_lr").
      setInputCols(featureColsLr).
      setOutputCol("features_lr")

    val estimators: Array[PipelineStage] = Array(continuousFeatureAssembler, continuousFeatureScaler, polyExpansionAssembler, continuousFeaturePolynomialExpansion).
      union(categoricalFeatureIndexers).
      union(categoricalFeatureOneHotEncoders).
      union(Seq(featureAssemblerLr))

    val featurePipeline = new Pipeline(uid = "feature_pipeline").
      setStages(estimators)
    val sparkFeaturePipelineModel = featurePipeline.fit(datasetImputedFiltered)

    //Step 8: Train Logistic Regression Model
    val logisticRegression = new LogisticRegression(uid = "logistic_regression").
      setFeaturesCol("features_lr").
      setLabelCol("approved").
      setPredictionCol("approved_prediction")

    val sparkPipelineEstimatorLr = new Pipeline().setStages(Array(sparkFeaturePipelineModel, logisticRegression))
    val sparkPipelineLr = sparkPipelineEstimatorLr.fit(datasetImputedFiltered)

    println("Complete: Training Logistic Regression")

    //Step 9: (Optional): Serialize your models to bundle.ml
    val sbc = SparkBundleContext().withDataset(sparkPipelineLr.transform(datasetImputedFiltered))
    for(bf <- managed(BundleFile("jar:file:/tmp/lc.model.lr.zip"))) {
      sparkPipelineLr.writeBundle.save(bf)(sbc).get
    }
  }
}