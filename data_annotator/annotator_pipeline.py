import asyncio
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor

from datasets import Dataset, DatasetDict

from base_annotator import DataAnnotator
import os


class AnnotationExecutor:
    def __init__(
        self, annotator_class: DataAnnotator, num_workers: int = os.cpu_count()
    ):
        self.annotator_class = annotator_class
        self.num_workers = num_workers

    async def run(self, dataset_name: str, save_path: str, **annotator_kwargs):
        """Execute annotation across multiple processes"""
        dataset = self.annotator_class(**annotator_kwargs).load_data(dataset_name)
        splits = self.annotator_class.detect_splits(dataset)

        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(
                    executor,
                    self._process_split,
                    self.annotator_class,
                    dataset[split],
                    split,
                    save_path,
                    annotator_kwargs,
                )
                for split in splits
            ]
            await asyncio.gather(*tasks)

    def _process_split(
        self,
        annotator_class: DataAnnotator,
        split_data: Dataset,
        split_name: str,
        save_path: str,
        annotator_kwargs: Dict,
    ):
        """Process a single split in a separate process"""
        annotator = annotator_class(**annotator_kwargs)
        processed_split = asyncio.run(annotator.process_split(split_data))
        annotated_split = split_data.add_column("llm_annotation", processed_split)
        annotated_split.save_to_disk(f"{save_path}_{split_name}")


class AnnotatorPipeline:
    def __init__(self, annotators: List[DataAnnotator]):
        """
        Initialize pipeline with list of annotators to apply sequentially
        Each annotator should use a unique annotation column name
        """
        self.annotators = annotators
        self.executors = [
            AnnotationExecutor(annotator.__class__) for annotator in self.annotators
        ]

    async def run_pipeline(
        self, dataset_name: str, save_path: str, **annotator_kwargs
    ) -> DatasetDict:
        """
        Execute multiple annotators sequentially on the dataset
        Merges all annotations into final dataset
        """
        # Load initial dataset
        initial_dataset = self.annotators[0].load_data(dataset_name)
        current_dataset = initial_dataset

        # Apply each annotator sequentially
        for annotator, executor in zip(self.annotators, self.executors):
            combined_kwargs = {**annotator.__dict__, **annotator_kwargs}

            # Process dataset with current annotator
            current_dataset = await executor.process_dataset(
                current_dataset, **combined_kwargs
            )

        # Save final combined dataset
        current_dataset.save_to_disk(save_path)
        return current_dataset
