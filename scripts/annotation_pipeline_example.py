import asyncio
import sys
sys.path.append("..")

from data_annotator.annotator_pipeline import AnnotatorPipeline

from dotenv import load_dotenv
load_dotenv()

DATASET_NAME = "RAGEVALUATION-HJKMY/ragbench_10row_tester"  

from data_annotator.annotators import KeyPointAnnotator

async def main():
    pipeline = AnnotatorPipeline([KeyPointAnnotator])
    await pipeline.run_pipeline(
        dataset_name= DATASET_NAME,
        save_path= "./tmp_data",
        upload_to_hub= True,
        repo_id="RAGEVALUATION-HJKMY/ragbench_10row_tester_annotated"
    )
if __name__ == "__main__":
    asyncio.run(main())