from langchain_core.pydantic_v1 import BaseModel

from ragas.llms.output_parser import RagasoutputParser, get_json_format_instructions
from ragas.llms.prompt import Prompt


class AnswerFormat(BaseModel):
    answer: str
    verdict: int


question_answer_parser = RagasoutputParser(pydantic_object=AnswerFormat)

# 推理问题提示
reasoning_question_prompt = Prompt(
    name="reasoning_question",
    instruction="""根据提供的上下文将问题重写为多跳推理问题，使给定的问题复杂化。
    回答问题应要求读者使用给定上下文中可用的信息进行多重逻辑连接或推理。
    重写问题时应遵循的规则：
    1. 确保重写的问题可以完全根据上下文中存在的信息来回答。
    2. 不要构造包含超过 15 个单词的问题。尽可能使用缩写。
    3. 确保问题清晰无歧义。
    4. 问题中不允许出现“基于提供的上下文”、“根据上下文”等短语。""",
    examples=[
        {
            "question": "法国的首都是哪里？",
            "context": "法国是西欧的一个国家。它有巴黎、里昂和马赛等多座城市。巴黎不仅以埃菲尔铁塔和卢浮宫等文化地标而闻名，也是法国的行政中心。",
            "output": "哪座城市同时连接埃菲尔铁塔和行政中心？",
        },
        {
            "question": "Python 中 append() 方法起什么作用？",
            "context": "在 Python 中，列表用于将多个项目存储在单个变量中。列表是用于存储数据集合的 4 种内置数据类型之一。append() 方法将单个项目添加到列表末尾。",
            "output": "如果列表代表变量集合，那么哪种方法可以将其扩展一项？",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="chinese",
)

# 多上下文问题提示
multi_context_question_prompt = Prompt(
    name="multi_context_question",
    instruction="""
    任务是重写并复杂化给定的问题，以便回答它需要从上下文 1 和上下文 2 中获取的信息。
    重写问题时请遵循以下规则。
        1. 重写的问题不应太长。尽可能使用缩写。
        2. 重写的问题必须合理，必须能被人类理解和回答。
        3. 重写的问题必须能够从上下文 1 和上下文 2 中提供的信息中完全回答。
        4. 阅读并理解两个上下文，然后重写问题，以便回答需要从上下文 1 和上下文 2 中获得见解。
        5. 问题中不允许出现“基于提供的上下文”、“根据上下文？”等短语。""",
    examples=[
        {
            "question": "什么过程使植物变绿？",
            "context1": "叶绿素是一种使植物呈现绿色并帮助它们进行光合作用的色素。",
            "context2": "植物的光合作用通常发生在叶绿体集中的叶子中。",
            "output": "在哪些植物结构中，负责其青翠的色素有助于能量产生？",
        },
        {
            "question": "如何计算矩形的面积？",
            "context1": "图形的面积是根据图形的尺寸来计算的。对于矩形，这需要将长度乘以宽度。",
            "context2": "长方形有四条边，且对边的长度相等。长方形是四边形的一种。",
            "output": "哪些涉及相等对立数的乘法可以得出四边形的面积？",
        },
    ],
    input_keys=["question", "context1", "context2"],
    output_key="output",
    output_type="str",
    language="chinese",
)

# 条件问题提示
conditional_question_prompt = Prompt(
    name="conditional_question",
    instruction="""通过引入条件元素来重写提供的问题以增加其复杂性。
    目标是通过合并影响问题上下文的场景或条件使问题更加复杂。
    重写问题时请遵循以下规则。
        1. 重写的问题不应超过 25 个字。尽可能使用缩写。
        2. 重写的问题必须合理，必须能被人类理解和回答。
        3. 重写的问题必须能够根据当前上下文信息完全回答。
        4. 问题中不允许出现“提供的上下文”、“根据上下文？”等短语。""",
    examples=[
        {
            "question": "植物的根的作用是什么？",
            "context": "植物的根从土壤中吸收水分和养分，将植物固定在地面上，并储存养分。",
            "output": "植物根部对土壤养分和稳定性有何双重作用？",
        },
        {
            "question": "疫苗如何预防疾病？",
            "context": "疫苗通过刺激人体的免疫反应产生抗体来识别和对抗病原体，从而预防疾病。",
            "output": "疫苗如何利用人体免疫系统来防御病原体？",
        },
    ],
    input_keys=["question", "context"],
    output_key="output",
    output_type="str",
    language="chinese",
)

# 压缩问题提示
compress_question_prompt = Prompt(
    name="compress_question",
    instruction="""重写以下问题，使其更间接、更简短，同时保留原始问题的本质。
    目标是创建一个以不太直接的方式传达相同含义的问题。重写的问题应该更短，因此尽可能使用缩写。""",
    examples=[
        {
            "question": "地球和月球之间的距离是多少？",
            "output": "月球距离地球有多远？",
        },
        {
            "question": "烘烤巧克力蛋糕需要哪些原料？",
            "output": "制作巧克力蛋糕需要什么？",
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="str",
    language="chinese",
)

# 对话式问题提示
conversational_question_prompt = Prompt(
    name="conversation_question",
    instruction="""将提供的问题重新格式化为两个单独的问题，就像它是对话的一部分一样。每个问题都应关注与原始问题相关的特定方面或子主题。
    重写问题时请遵循以下规则。
        1. 重写的问题不应超过 25 个字。尽可能使用缩写。
        2. 重写的问题必须合理，必须能被人类理解和回答。
        3. 重写的问题必须能够根据当前上下文信息完全回答。
        4. 问题中不允许出现“提供的上下文”、“根据上下文？”等短语。""",
    examples=[
        {
            "question": "远程工作的优点和缺点是什么？",
            "output": {
                "first_question": "远程工作有什么好处？",
                "second_question": "另一方面，远程工作会遇到哪些挑战？",
            },
        }
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="chinese",
)

# 问题答案提示
question_answer_prompt = Prompt(
    name="answer_formulate",
    instruction="""使用给定上下文中的信息回答问题。如果答案存在，则输出结论'1'；如果答案不存在，则输出结论'-1'。""",
    output_format_instruction=get_json_format_instructions(AnswerFormat),
    examples=[
        {
            "context": """气候变化很大程度上受到人类活动的影响，特别是燃烧化石燃料产生的温室气体。大气中温室气体浓度的增加会吸收更多的热量，导致全球变暖和天气模式的变化。""",
            "question": "人类活动如何导致气候变化？",
            "answer": AnswerFormat.parse_obj(
                {
                    "answer": "人类活动主要通过燃烧化石燃料排放温室气体来导致气候变化。这些排放增加了大气中温室气体的浓度，从而吸收更多热量，导致全球变暖和天气模式改变。",
                    "verdict": "1",
                }
            ).dict(),
        },
        {
            "context": """人工智能 (AI) 的概念随着时间的推移而不断发展，但它本质上是指旨在模仿人类认知功能的机器。人工智能可以像人类一样学习、推理、感知，在某些情况下还能做出反应，这使得它在从医疗保健到自动驾驶汽车等领域发挥着关键作用。""",
            "question": "人工智能的关键能力有哪些？",
            "answer": AnswerFormat.parse_obj(
                {
                    "answer": "人工智能旨在模仿人类的认知功能，其关键能力包括学习、推理、感知以及以类似人类的方式对环境做出反应。这些能力使人工智能在医疗保健和自动驾驶等各个领域发挥着重要作用。",
                    "verdict": "1",
                }
            ).dict(),
        },
        {
            "context": """简·奥斯汀的小说《傲慢与偏见》围绕伊丽莎白·班纳特及其家人展开。故事背景设定在 19 世纪的英国乡村，涉及婚姻、道德和误解等问题。""",
            "question": "《傲慢与偏见》出版于哪一年？",
            "answer": AnswerFormat.parse_obj(
                {
                    "answer": "给定问题的答案在上下文中不存在",
                    "verdict": "-1",
                }
            ).dict(),
        },
    ],
    input_keys=["context", "question"],
    output_key="answer",
    output_type="json",
    language="chinese",
)

# 关键词提取提示
keyphrase_extraction_prompt = Prompt(
    name="keyphrase_extraction",
    instruction="从提供的文本中提取前 3 到 5 个关键短语，重点关注最重要和最独特的方面。",
    examples=[
        {
            "text": "黑洞是时空中的一个区域，这里的引力非常强大，以至于包括光和其他电磁波在内的任何东西都没有足够的能量逃脱。广义相对论预测，足够致密的质量可以使时空变形，形成黑洞。",
            "output": {
                "keyphrases": [
                    "黑洞",
                    "时空区域",
                    "强引力",
                    "光和电磁波",
                    "广义相对论",
                ]
            },
        },
        {
            "text": "长城是位于中国北方的一系列古代城墙和防御工事，建于大约 500 年前。这堵巨大的城墙绵延 13,000 多英里，是古代中国工程师的技术和毅力的见证。",
            "output": {
                "keyphrases": [
                    "中国长城",
                    "古代防御工事",
                    "中国北方",
                ]
            },
        },
    ],
    input_keys=["text"],
    output_key="output",
    output_type="json",
    language="chinese",
)

# 种子问题提示
seed_question_prompt = Prompt(
    name="seed_question",
    instruction="生成一个可以从给定上下文中完全回答的问题。问题应使用主题形成",
    examples=[
        {
            "context": "植物的光合作用涉及将光能转化为化学能，利用叶绿素和其他色素吸收光。这一过程对于植物生长和氧气的产生至关重要。",
            "keyphrase": "光合作用",
            "question": "光合作用在植物生长中起什么作用？",
        },
        {
            "context": "始于18世纪的工业革命标志着历史的一个重大转折点，因为它推动了工厂和城市化的发展。",
            "keyphrase": "工业革命",
            "question": "工业革命如何成为历史的重大转折点？",
        },
        {
            "context": "蒸发过程在水循环中起着至关重要的作用，它将水从液体转化为蒸汽，并使其上升到大气中。",
            "keyphrase": "蒸发",
            "question": "为什么蒸发在水循环中很重要？",
        },
    ],
    input_keys=["context", "keyphrase"],
    output_key="question",
    output_type="str",
    language="chinese",
)

# 主要话题提取提示
main_topic_extraction_prompt = Prompt(
    name="main_topic_extraction",
    instruction="识别并提取给定文本中深入讨论的两个主要主题。",
    examples=[
        {
            "text": "区块链技术提供了一种去中心化账本，可确保数据交易的完整性和透明度。它支撑着比特币等加密货币，为所有交易提供安全且不可篡改的记录。除了金融之外，区块链在供应链管理方面也有潜在的应用，可以简化运营、增强可追溯性并改善欺诈预防。它允许实时跟踪货物并在参与者之间透明地共享数据。",
            "output": {
                "topics": [
                    "区块链技术及其在加密货币中的基础作用",
                    "区块链在供应链管理中的应用",
                ]
            },
        },
        {
            "text": "远程医疗彻底改变了医疗保健的提供方式，尤其是在农村和服务欠缺的地区。它允许患者通过视频会议咨询医生，改善医疗服务，减少出行需求。医疗保健的另一项重大进步是精准医疗，即根据个人基因特征量身定制治疗方案。这种方法为多种疾病（包括某些癌症和慢性病）带来了更有效的治疗方法。",
            "output": {
                "topics": [
                    "远程医疗及其对医疗保健可及性的影响",
                    "精准医疗及其在根据基因特征定制治疗方案中的作用",
                ]
            },
        },
    ],
    input_keys=["text"],
    output_key="output",
    output_type="json",
    language="chinese",
)

# 查找相关上下文提示
find_relevant_context_prompt = Prompt(
    name="find_relevant_context",
    instruction="给定一个问题和一组上下文，找到最相关的上下文来回答该问题。",
    examples=[
        {
            "question": "法国的首都是哪里？",
            "contexts": [
                "1. 法国是西欧的一个国家。它有巴黎、里昂和马赛等几座城市。巴黎不仅以埃菲尔铁塔和卢浮宫等文化地标而闻名，而且还是法国的行政中心。",
                "2. 法国的首都是巴黎。它也是法国人口最多的城市，人口超过 200 万。巴黎以埃菲尔铁塔和卢浮宫等文化地标而闻名。",
                "3. 巴黎是法国的首都。它也是法国人口最多的城市，人口超过 200 万。巴黎以埃菲尔铁塔和卢浮宫等文化地标而闻名。",
            ],
            "output": {
                "relevant_contexts": [1, 2],
            },
        },
        {
            "question": "咖啡因对身体有何影响？其常见来源有哪些？",
            "contexts": [
                "1. 咖啡因是一种中枢神经系统兴奋剂。它可以暂时消除困倦并恢复警觉。它主要影响大脑，改变神经递质的功能。",
                "2. 定期进行体育锻炼对于保持身体健康至关重要。它可以帮助控制体重、对抗健康状况、增强能量并促进更好的睡眠。",
                "3. 咖啡因的常见来源包括咖啡、茶、可乐和能量饮料。这些饮料在世界各地都有消费，并以快速补充能量而闻名。",
            ],
            "output": {"relevant_contexts": [1, 2]},
        },
    ],
    input_keys=["question", "contexts"],
    output_key="output",
    output_type="json",
    language="chinese",
)

# 问题重写提示
question_rewrite_prompt = Prompt(
    name="rewrite_question",
    instruction="""给定背景、问题和反馈，根据提供的反馈重写问题以提高其清晰度和可回答性。""",
    examples=[
        {
            "context": "埃菲尔铁塔是用铁建造的，最初是作为 1889 年巴黎世界博览会的临时展品。尽管最初只是临时用途，但埃菲尔铁塔很快成为巴黎独创性的象征和这座城市的标志性地标，每年吸引数百万游客。这座由古斯塔夫·埃菲尔 (Gustave Eiffel) 设计的铁塔最初遭到了一些法国艺术家和知识分子的批评，但后来被誉为结构工程和建筑设计的杰作。",
            "question": "谁创造了铁塔的设计？",
            "feedback": "问题询问了'铁塔'设计的创造者，但没有具体说明它指的是哪座塔。世界上有很多塔，如果不指定确切的塔，这个问题就不清楚，无法回答。为了改进这个问题，它应该包括所讨论的特定塔的名称或清晰的描述。",
            "output": "谁创造了埃菲尔铁塔的设计？",
        },
        {
            "context": "《探索神经网络中的零样本学习》由 Smith 和 Lee 于 2021 年出版，重点关注零样本学习技术在人工智能中的应用。",
            "question": "本研究中的零样本评估使用了哪些数据集？",
            "feedback": "“问题询问'本研究'中用于零样本评估的数据集，而没有具体说明或提供有关该研究的任何详细信息。这使得那些无法访问或不了解具体研究的人无法理解问题。为了提高清晰度和可回答性，问题应该具体说明它所指的研究，或者提供足够的研究背景，以便理解和独立回答问题。",
            "output": "在神经网络中探索零样本学习的论文中，零样本评估使用了哪些数据集？",
        },
    ],
    input_keys=["context", "question", "feedback"],
    output_key="output",
    output_type="str",
    language="chinese",
)

### Filters


class ContextScoring(BaseModel):
    clarity: int
    depth: int
    structure: int
    relevance: int


class QuestionFilter(BaseModel):
    feedback: str
    verdict: int


class EvolutionElimination(BaseModel):
    reason: str
    verdict: int


context_scoring_parser = RagasoutputParser(pydantic_object=ContextScoring)
question_filter_parser = RagasoutputParser(pydantic_object=QuestionFilter)
evolution_elimination_parser = RagasoutputParser(pydantic_object=EvolutionElimination)

# 情境评分提示
context_scoring_prompt = Prompt(
    name="score_context",
    instruction="""
    给定一个上下文，执行以下任务并以有效的 JSON 格式输出答案：评估所提供的上下文，并为您的 JSON 响应中的以下每个标准分配 1（低）、2（中）或 3（高）的数字分数：

清晰度：评估所呈现信息的精确度和可理解性。高分（3）适用于信息准确且易于理解的上下文。低分（1）适用于信息模糊或难以理解的上下文。
深度：确定详细检查的水平以及上下文中创新见解的包含程度。高分表示全面而有见地的分析，而低分则表示对主题的处理肤浅。
结构：评估内容的组织程度以及是否合乎逻辑。高分授予展示连贯组织和逻辑进展的上下文，而低分则表示缺乏结构或进展清晰度。
相关性：判断内容与主题的相关性，对紧密围绕主题且没有不必要离题的内容给予高分，对充斥着无关信息的上下文给予低分。
构建 JSON 输出，将这些标准反映为键，将其对应的分数反映为值""",
    output_format_instruction=get_json_format_instructions(ContextScoring),
    examples=[
        {
            "context": "勾股定理是几何学中的一个基本原理。它指出，在直角三角形中，斜边（直角对边）长度的平方等于其他两边长度的平方和。这可以写成 a^2 + b^2 = c^2，其中 c 表示斜边的长度，a 和 b 表示其他两边的长度。",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 1, "structure": 3, "relevance": 3}
            ).dict(),
        },
        {
            "context": "阿尔伯特·爱因斯坦（1879 年 3 月 14 日 - 1955 年 4 月 18 日）是一位德国出生的理论物理学家，被广泛认为是有史以来最伟大、最具影响力的科学家之一。",
            "output": ContextScoring.parse_obj(
                {"clarity": 3, "depth": 2, "structure": 3, "relevance": 3}
            ).dict(),
        },
        {
            "context": "我喜欢巧克力。它真的很好吃。哦，顺便说一句，地球绕着太阳转，而不是太阳绕地球转。另外，我最喜欢的颜色是蓝色。",
            "output": ContextScoring.parse_obj(
                {"clarity": 2, "depth": 1, "structure": 1, "relevance": 1}
            ).dict(),
        },
    ],
    input_keys=["context"],
    output_key="output",
    output_type="json",
    language="chinese",
)

# 过滤问题提示
filter_question_prompt = Prompt(
    name="filter_question",
    instruction="""
在具备足够的领域知识的情况下，评估给定问题的清晰度和可回答性，考虑以下标准：
1.独立性：是否可以理解和回答问题，而无需额外的背景信息或访问问题本身未提供的外部参考资料？问题应该是独立的，这意味着它们不依赖于问题中未共享的特定文档、表格或先前知识。
2.明确的意图：问题寻求的答案或信息类型是否清楚？问题应明确传达其目的，允许直接和相关的回答。
根据这些标准，如果问题具体、独立且意图明确，可根据提供的详细信息理解和回答，则分配"1"的判定。如果由于含糊不清、依赖外部参考资料或意图模糊而未能满足这些标准中的一个或多个，则分配"0"。
以 JSON 格式提供反馈和判定，如果问题被认为不清楚，则包括改进建议。突出问题中哪些方面导致问题的清晰度或不清晰度，并就如何重新构建或详细阐述问题以便更好地理解和回答提供建议。
""",
    output_format_instruction=get_json_format_instructions(QuestionFilter),
    examples=[
        {
            "question": "关于太空有什么发现？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "这个问题太过模糊和宽泛，要求回答“关于太空的发现”，但没有具体说明感兴趣的任何方面、时间范围或背景。这可能涉及广泛的主题，从发现新的天体到太空旅行技术的进步。为了提高清晰度和可答性，问题可以具体说明发现的类型（例如天文、技术）、时间范围（例如近期、历史）或背景（例如在特定研究或太空任务中）。",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "根据 context1 和 context2 的结果，ALMA-13B-R 与 WMT'23 研究中的其他翻译模型相比表现如何？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "此问题要求将 ALMA-13B-R 模型的性能与 WMT'23 研究中的其他翻译模型进行比较，具体指'context1'和'context2'中的结果。虽然它明确指定了感兴趣的模型 (ALMA-13B-R) 和研究 (WMT'23)，但它假设可以访问和理解'context1'和'context2'，但没有解释这些上下文的含义。这使得那些不熟悉 WMT'23 研究或这些特定上下文的人无法理解这个问题。为了提高更广泛受众的清晰度和可回答性，定义或描述'context1'和'context2'，或解释在这些上下文中用于比较的标准，可能会对这个问题有所帮助。",
                    "verdict": "0",
                }
            ).dict(),
        },
        {
            "question": "KIWI-XXL 和 XCOMET 在评估分数、翻译模型性能以及超越参考文献的成功率方面与表 1 中的黄金标准参考文献相比如何？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "问题要求比较 KIWI-XXL 和 XCOMET 模型与'表 1'中的黄金标准参考文献，重点关注评估分数、翻译模型性能以及超越参考文献的成功率。它指定了比较的模型和标准，使意图清晰。但是，问题假设可以访问'表 1'，但不提供其内容或上下文，这对于那些无法直接访问源材料的人来说是不清楚的。为了让普通观众更清楚、更容易回答，问题可以包括'表 1'的内容或主要发现的简要描述，或者以不依赖特定未发表文件的方式来构建问题。",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question": "OpenMoE 中 UL2 训练目标的配置是怎样的，为什么它是预训练的更好选择？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "问题询问 OpenMoE 框架内 UL2 训练目标的配置及其适合预训练的理由。它明确指定了感兴趣的主题（UL2 训练目标，OpenMoE），并寻求有关配置及其在预训练中有效的原因的详细信息。但是，对于那些不熟悉特定术语或 OpenMoE 和 UL2 背景的人来说，这个问题可能具有挑战性。为了更清晰和可回答性，如果问题包含有关 OpenMoE 和 UL2 训练目标的简要说明或背景，或者澄清它所指的预训练有效性的各个方面（例如效率、准确性、泛化），将会很有帮助。",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question": "根据提供的上下文，OpenMoE 中 UL2 训练目标的详细配置是什么？",
            "output": QuestionFilter.parse_obj(
                {
                    "feedback": "该问题寻求有关 OpenMoE 框架内 UL2 训练目标配置的详细信息，其中提到了“提供的上下文”，但实际上并未在查询中包含或描述此上下文。这会让那些无法访问未指定上下文的人无法理解这个问题。为了使问题清晰易答，它需要直接在问题中包含相关上下文，或者以不需要外部信息的方式进行构建。详细说明感兴趣的配置的具体方面（例如，损失函数、数据增强技术）也有助于澄清查询。",
                    "verdict": 0,
                }
            ).dict(),
        },
    ],
    input_keys=["question"],
    output_key="output",
    output_type="json",
    language="chinese",
)

# 进化消除提示
evolution_elimination_prompt = Prompt(
    name="evolution_elimination",
    instruction="""根据以下要求检查给定的两个问题是否相等：
    1. 它们具有相同的约束和要求。
    2. 它们具有相同的探究深度和广度。
    如果它们相等，则输出判定结果 1，否则输出 0""",
    output_format_instruction=get_json_format_instructions(EvolutionElimination),
    examples=[
        {
            "question1": "气候变化的主要原因是什么？",
            "question2": "哪些因素造成全球变暖？",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "虽然这两个问题都涉及环境问题，但“气候变化”比“全球变暖”涵盖的变化更为广泛，因此探究的深度也不同。",
                    "verdict": 0,
                }
            ).dict(),
        },
        {
            "question1": "植物的光合作用如何进行？",
            "question2": "你能解释一下植物的光合作用过程吗？",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "这两个问题都要求对植物的光合作用过程进行解释，其答案的深度、广度和要求相同。",
                    "verdict": 1,
                }
            ).dict(),
        },
        {
            "question1": "定期锻炼对健康有哪些益处？",
            "question2": "你能列举一下经常锻炼对健康的好处吗？",
            "output": EvolutionElimination.parse_obj(
                {
                    "reason": "这两个问题都寻求有关定期锻炼对健康的积极影响的信息。它们要求以类似的详细程度列出健康益处。",
                    "verdict": 1,
                }
            ).dict(),
        },
    ],
    input_keys=["question1", "question2"],
    output_key="output",
    output_type="json",
    language="chinese",
)

testset_prompts = [
    reasoning_question_prompt,
    multi_context_question_prompt,
    conditional_question_prompt,
    compress_question_prompt,
    conversational_question_prompt,
    question_answer_prompt,
    keyphrase_extraction_prompt,
    seed_question_prompt,
    main_topic_extraction_prompt,
    find_relevant_context_prompt,
    question_rewrite_prompt,
    context_scoring_prompt,
    filter_question_prompt,
    evolution_elimination_prompt,
]
