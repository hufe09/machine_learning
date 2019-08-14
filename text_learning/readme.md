#  文本学习

## 词袋 Bag of Words

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.r0lctkhkkyp.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.2a2b3j55hb.png)

```
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()

string1 = "Remove accents and perform other character normalization during the preprocessing step."
string2 = "Convert all characters to lowercase before tokenizing."
string3 = "Override the preprocessing (string transformation) stage while n-grams generation steps."

email_list = [string1, string2, string3]
bag_of_words = vectorizer.fit_transform(email_list)
print(vectorizer.get_feature_names())
bag_of_words = vectorizer.transform(email_list)
bag_of_words.toarray()
```

```
['accents', 'all', 'and', 'before', 'character', 'characters', 'convert', 'during', 'generation', 'grams', 'lowercase', 'normalization', 'other', 'override', 'perform', 'preprocessing', 'remove', 'stage', 'step', 'steps', 'string', 'the', 'to', 'tokenizing', 'transformation', 'while']
array([[1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1,
        0, 0, 0, 0],
       [0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 1, 0, 0],
       [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1,
        0, 0, 1, 1]])
```

## 停用词 Stop Words
```
# pip install NLTK
# import nltk
# nltk.download("stopwords")
```

```
from nltk.corpus import stopwords
sw = stopwords.words("english")
len(sw)
```

```
179
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.io3bdd6ik4.png)

## 词干化以合并词汇

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.fr6stb94ol4.png)

### 提取器 Stemmer Snowball 

```
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

print(stemmer.stem("responsiveness"))
print(stemmer.stem("unresponsivity"))
```
```
'respons'
'unrespons'
```

## 文本处理中的运算符顺序
先构建词袋还是先构建词干。

- 在构建单词袋之前进行词干提取。



如果理解起来很抽象，此处的示例也许能帮到你：

假设我们正在讨论“responsibility is responsive to responsible people”这一段文字（这句话不合语法，但你知道我的意思……）

如果你直接将这段文字放入词袋，你得到的就是：
```
[is:1 
people: 1
responsibility: 1
responsive: 1
responsible:1]
```
然后再运用词干化，你会得到 
```
[is:1
people:1
respon:1
respon:1
respon:1]
```
(如果你可以找到方法在 sklearn 中词干化计数向量器的对象，这种尝试最有可能的结果就是你的代码会崩溃……)

那样，你就需要再进行一次后处理来得到以下词袋，如果你一开始就进行词干化，你就会直接获得这个词袋了：
```
[is:1 
people:1
respon:3]
```
显然，第二个词袋很可能是你想要的，所以在此处先进行词干化可使你获得正确的答案。



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.lvgvfmrc2v7.png)

# Mini-project

*tools/parse_out_email_text.py*

词干化之前：

`Hi Everyone  If you can read this message youre properly using parseOutText  Please proceed to the next part of the project`

## 词干化

去标点符号

```
text_string = content[1].translate(str.maketrans({key: None for key in string.punctuation}))
```

将每个单词词干化

```
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
print(words)

words_list = []
for word in words.split(" "):
    if word != "":
        stem = stemmer.stem(word)
        words_list.append(stem)
print(words_list)

res = " ".join(words_list)
```

`hi everyon if you can read this messag your proper use parseouttext pleas proceed to the next part of the project`



## 清除“签名文字”

*text_learning/vectorize_text.py*

在 *vectorize_text.py* 中，你将迭代所有来自 Chris 和 Sara 的邮件。 将每封已读邮件提供给 *parseOutText()* 并返回词干化的文本字符串。然后做以下两件事：

删除签名文字（“sara”、“shackleton”、“chris”、“germani”——如果你知道为什么是“germani”而不是“germany”，你将获得加分）

向 word_data 添加更新的文本字符串——如果邮件来自 Sara，向 from_data 添加 0（零），如果是 Chris 写的邮件，则添加 1。

完成此步骤后，你应该有两个列表：一个包含了每封邮件被词干化的正文，第二个应该包含用来编码（通过 0 或 1）谁是邮件作者的标签。

对所有邮件运行程序需要花一些时间（5 分钟或更长时间），所以我们添加了一个 temp_counter，将第 200 封之后的邮件切割掉。 当然，一切就绪后，你会希望对整个数据集运行程序。

```
for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        # only look at first 200 emails when developing
        # once everything is working, remove this line to run over full dataset
        path = os.path.join('../../../../', path[:-1])

        if not os.path.exists(path):
            path = os.path.join('../../../../', path[:-2] + "_")

        temp_counter += 1
        if temp_counter < 200:
            text = ""
            try:
                with open(path, "r") as email:
                    # use parseOutText to extract the text from the opened email
                    # text = parseOutText(email)
                    text = parseOutText(email)

            except Exception as e:
                print(e)

            # use str.replace() to remove any instances of the words
            # ["sara", "shackleton", "chris", "germani"]
            text = text.replace("sara", "").replace("shackleton", '').replace("chris", '').replace("germani", '')

            # append the text to word_data
            if text != "":
                word_data.append(text)

                # append a 0 to from_data if email is from Sara, and 1 if email is from Chris
                if name == "sara":
                    from_data.append(0)
                if name == "chris":
                    from_data.append(1)
```



## TfIdf vectorization

TF-IDF 是一个统计方法，用来评估某个词语对于一个文件集或文档库中的其中一份文件的重要程度。

TF-IDF 实际上是两个词组 Term Frequency 和 Inverse Document Frequency 的总称，两者缩写为 TF 和 IDF，分别代表了词频和逆向文档频率。

使用 sklearn TfIdf 转换将 word_data 转换为 tf-idf 矩阵。删除英文停止词。

你可以使用 *get_feature_names()* 访问单词和特征数字之间的映射，该函数返回一个包含词汇表所有单词的列表。

```
# in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

stop_words = stopwords.words("english")
tfidf_vec = TfidfVectorizer(stop_words=stop_words)
tfidf_matrix = tfidf_vec.fit_transform(word_data)
print('不重复的词:', tfidf_vec.get_feature_names())
print('每个单词的 ID:', tfidf_vec.vocabulary_)
print('每个单词的 tfidf 值:', tfidf_matrix.toarray())
```

```
不重复的词: ['abnorm', 'abov', 'absenc', 'ac', 'academi', ...]
每个单词的 ID: {'sbaile2': 2649, 'nonprivilegedpst': 2232, 'see': 2672, 'comment': 972,...}

每个单词的 tfidf 值: 
[[0.         0.         0.06498238 ... 0.         0.         0.        ]
 [0.         0.         0.06498238 ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]
 ...
 [0.         0.         0.         ... 0.         0.         0.        ]
 [0.         0.07243445 0.         ... 0.         0.         0.        ]
 [0.         0.         0.         ... 0.         0.         0.        ]]
```

