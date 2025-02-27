import asyncio
import sys
sys.path.append("..")

from execution_pipeline.execution_pipeline import ExecutionPipeline

from dotenv import load_dotenv
load_dotenv()

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

DATASET_NAME = "RAGEVALUATION-HJKMY/ragbench_10row_tester_annotated"

from evaluator.evaluators import KeyPointEvaluator, LearningFacilitationEvaluator, BERTScoreEvaluator

async def main():
    pipeline = ExecutionPipeline([BERTScoreEvaluator])
    await pipeline.run_pipeline(dataset_name=DATASET_NAME, save_path="./tmp_data", upload_to_hub=True,
                                repo_id="RAGEVALUATION-HJKMY/ragbench_10row_tester_evaluated")
if __name__ == "__main__":
    asyncio.run(main())