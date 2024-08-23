from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import shutil
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from custom import testset_prompts

# import nltk
# nltk.download('punkt')

load_dotenv()

if not os.path.exists("./documents"):
    raise Exception("documents 不存在")

# 尝试修复中文缓存格式问题
if os.path.exists("./cache"):
    shutil.rmtree("./cache")

loader = DirectoryLoader("./documents", loader_cls=TextLoader, show_progress=True)
documents = loader.load()

for document in documents:
    document.metadata['filename'] = document.metadata['source']

# generator with openai models
generator_llm = ChatOpenAI(model="qwen2:72b-instruct")
critic_llm = ChatOpenAI(model="qwen2:72b-instruct")
embeddings = OpenAIEmbeddings(model="bge-large-zh-v1.5", check_embedding_ctx_length=False, show_progress_bar=True)

# 加载中文提示词
for prompt in testset_prompts:
    prompt.save(cache_dir="./cache")

generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embeddings
)

# 使用翻译后的中文提示词
generator.adapt("chinese", evolutions=[simple, reasoning, multi_context, conditional], cache_dir="./cache")
generator.save(evolutions=[simple, reasoning, multi_context, conditional], cache_dir="./cache")
distributions = {simple: 0.5, reasoning: 0.2, multi_context: 0.2, conditional: 0.1}

# generate testset
testset = generator.generate_with_langchain_docs(documents, test_size=100, distributions=distributions, with_debugging_logs=True, raise_exceptions=False)

# 最终生成的测试集大小会小于设置的 test_size ，因为在 question_filter 的过程中会因为超过最大限制而忽略
testset.to_dataset().save_to_disk('./testset')
