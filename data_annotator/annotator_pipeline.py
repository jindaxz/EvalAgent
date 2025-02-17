import asyncio
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from datasets import Dataset, DatasetDict
from data_annotator.base_annotator import DataAnnotator
from data_annotator.annotators import KeyPointAnnotator
import os


class AnnotationExecutor:
    def __init__(
        self, annotator_class: type[DataAnnotator], num_workers: int = os.cpu_count()
    ):
        self.annotator_class = annotator_class
        self.num_workers = num_workers

    async def run(self, dataset: DatasetDict, **annotator_kwargs) -> DatasetDict:
        """Process entire DatasetDict across splits with parallel processing"""
        processed_splits = {}
        splits = self.annotator_class.detect_splits(dataset)

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._process_split,
                    self.annotator_class,
                    dataset[split],
                    annotator_kwargs,
                )
                for split in splits
            ]
            results = await asyncio.gather(*tasks)

        for split, result in zip(splits, results):
            processed_splits[split] = result

        return DatasetDict(processed_splits)

    @staticmethod
    def _process_split(
        annotator_class: type[DataAnnotator], split_data: Dataset, annotator_kwargs
    ):
        """Instantiate inside worker process"""
        annotator = annotator_class(**annotator_kwargs)  # Create instance here
        processed = asyncio.run(annotator.process_split(split_data))
        return split_data.add_column(annotator.annotation_column, processed)


class AnnotatorPipeline:
    def __init__(self, annotator_classes: List[type[DataAnnotator]]):
        self.annotator_classes = annotator_classes
        self.executors = [AnnotationExecutor(cls) for cls in annotator_classes]

    async def run_pipeline(
        self,
        dataset_name: str,
        save_path: str,
        upload_to_hub: bool = False,
        repo_id: Optional[str] = None,
        **annotator_kwargs
    ) -> DatasetDict:
        # Load initial dataset
        initial_dataset = self.annotator_classes[0].load_data(dataset_name)
        current_dataset = initial_dataset

        # Create fresh instances in executor processes
        for annotator_cls, executor in zip(self.annotator_classes, self.executors):
            current_dataset = await executor.run(
                dataset=current_dataset,
                annotator_class=annotator_cls,
                **annotator_kwargs
            )

        current_dataset.save_to_disk(save_path)

        # Upload to Hub if requested
        if upload_to_hub:
            if not repo_id:
                raise ValueError("repo_id is required for Hub upload")

            current_dataset.push_to_hub(repo_id=repo_id, token=os.getenv("HF_TOKEN"))

        return current_dataset
