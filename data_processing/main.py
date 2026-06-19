from tennis_pipeline import TennisPipeline

pipeline = TennisPipeline()

x, y = pipeline.execute()

pipeline.export(x, y)
