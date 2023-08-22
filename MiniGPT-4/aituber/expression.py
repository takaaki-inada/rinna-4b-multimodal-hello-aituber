import os

from mlask import MLAsk

os.environ["MECABRC"] = "/etc/mecabrc"

default_expression = 'neutral'
expression_array = [
    ['yorokobi', 'happy'],
    ['ikari', 'angry'],
    ['aware', 'sad'],
    ['kowagari', default_expression],
    ['haji', default_expression],
    ['suki', 'happy'],
    ['iya', 'sad'],
    ['takaburi', default_expression],
    ['yasuragi', 'happy'],
    ['odoroki', default_expression],
]
expression_dict = {key: value for key, value in expression_array}
# emotion_analyzer = MLAsk()
emotion_analyzer = MLAsk('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')


def get_expression(text: str) -> str:
    result = emotion_analyzer.analyze(text)
    # print(result)
    expression = default_expression
    if result.get("representative"):
        expression = expression_dict.get(result["representative"][0], default_expression)
        # print(expression)
    return expression


if __name__ == "__main__":
    texts = [
        '彼のことは嫌いではない',
        'しあわせ',
        '困った',
    ]
    for text in texts:
        print(get_expression(text))
