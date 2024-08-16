from src.dimond_price_prediction.pipelines.training_pipeline import TrainingPipeline


if __name__ == '__main__':
    training_pipeline = TrainingPipeline()
    training_pipeline.train()