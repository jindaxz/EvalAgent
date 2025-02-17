import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict, load_dataset
from utils.llm import LLMClient, OpenAIClientLLM
from data_annotator.prompt_manager import AnnotatePromptManager


class DataAnnotator(ABC):
    def __init__(
        self,
        llm_class: type[LLMClient] = None,
        annotation_column: str = "llm_annotation",
        **llm_kwargs
    ):
        self.llm = llm_class() if llm_class else OpenAIClientLLM()
        self.prompt_manager = AnnotatePromptManager()
        self.annotation_column = annotation_column

    @classmethod
    def load_data(cls, dataset_name: str, config: Optional[str] = None) -> DatasetDict:
        """Load dataset from Hugging Face hub"""
        dataset = load_dataset(dataset_name, config)
        if not isinstance(dataset, DatasetDict):
            dataset = DatasetDict({"train": dataset})
        return dataset

    @classmethod
    def detect_splits(cls, dataset: DatasetDict) -> List[str]:
        """Detect available splits in the dataset"""
        return [split for split in ["train", "validation", "test"] if split in dataset]

    async def annotate(
        self, dataset: DatasetDict, save_path: Optional[str] = None
    ) -> DatasetDict:
        """Modified to optionally return dataset instead of saving"""
        splits = self.detect_splits(dataset)
        processed = {}

        for split in splits:
            split_dataset = dataset[split]
            processed_split = await self.process_split(split_dataset)
            processed[split] = split_dataset.add_column(
                self.annotation_column, processed_split
            )

        annotated_dataset = DatasetDict(processed)
        if save_path:
            annotated_dataset.save_to_disk(save_path)
        return annotated_dataset

    async def process_split(self, split_dataset: Dataset) -> List[Dict]:
        """Process a single split asynchronously"""
        semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        futures = [self.process_row(row, semaphore) for row in split_dataset]
        return await asyncio.gather(*futures)

    async def process_row(self, row: Dict, semaphore: asyncio.Semaphore) -> Dict:
        """Process a single example with rate limiting"""
        async with semaphore:
            processed = self.pre_process(row)
            response = await self.call_llm(processed)
            return self.post_process(response, row)

    @abstractmethod
    def pre_process(self, example: Dict) -> Dict:
        """Format the example into a prompt"""
        pass

    @abstractmethod
    async def call_llm(self, processed_dict: Dict) -> str:
        """Call LLM with formatted prompt"""
        pass

    @abstractmethod
    def post_process(self, response: str, example: Dict) -> Dict:
        """Process LLM response into final format"""
        pass
