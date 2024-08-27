# GRAPH RAG

## Index

```bash
python -m graphrag.index --root .
```

## Query

### local

```bash
python -m graphrag.query --root . --method local "Q"
```

## Example

### RawText

```text
我是一段测试文本，只为了测试一个小段内容。
我的名字叫粽子，来自中国，我今年18岁。
我喜欢苹果。
我有一个朋友叫做张三，他来自印度，他比我大3岁。
张三的老婆叫李四，她和我同岁。
张三在学校里是我的学长，我在浙江大学就读。
张三喜欢吃苹果派，他老婆喜欢吃香蕉派。
粽子在学校里的射箭比赛获得了第三名，张三获得了第一名。
在夏季举行的围棋比赛中，粽子获得了第一名。
```

### Q&A 1

```bash
python -m graphrag.query --root . --method local "张三几岁？"

SUCCESS: Local Search Response: 根据提供的信息，我们知道张三来自印度，并且比王五大3岁。因此，我们可以推断出张三的年龄是18岁 + 3岁 = **21岁**。

数据参考：[Data: Entities (4); Relationships (4)]
```

### Q&A 2

```bash
python -m graphrag.query --root . --method local "浙江大学射箭比赛冠军是谁？"

SUCCESS: Local Search Response: 在提供的数据中，我们找到了关于射箭比赛的胜利者信息。根据"关系（Relationships）"表中的记录2和4，我们可以得知：

- "张三在学校里的射箭比赛获得了第一名。"
- "粽子在学校里的射箭比赛获得了第三名。"

因此，浙江大学射箭比赛的冠军是**张三**。

支持数据来源：[Data: Relationships (2, 4)]
```
