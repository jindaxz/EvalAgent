from enum import Enum


class BasePrompt(Enum):
    """Base class for prompt enums with template and output formatting"""

    @property
    def template(self) -> str | callable:
        return self.value['template']

    @property
    def criteria(self) -> str | callable:
        return self.value.get('criteria', '')

    @property
    def formatter(self) -> str | callable:
        return self.value['formatter']

    @property
    def examples(self) -> str | callable:
        return self.value.get('examples', '')

    @classmethod
    def get_prompt_type(cls, name: str) -> 'BasePrompt':
        return cls[name.upper()]
