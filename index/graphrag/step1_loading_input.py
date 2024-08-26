import asyncio
import os

from graphrag.index.input import load_input
from graphrag.index.config import PipelineTextInputConfig

sample_data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "input"
)

async def run_():
    dataset = await load_input(
        PipelineTextInputConfig(file_pattern=".*\\.txt$", base_dir=sample_data_dir)
    )
    print(dataset)

if __name__ == "__main__":
    asyncio.run(run_())
