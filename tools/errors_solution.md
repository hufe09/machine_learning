# 排错记录

```
from sklearn import cross_validation

cross_validation.train_test_split
```

已经不能用，用下面的`model_selection`替换

```
from sklearn.model_selection import train_test_split
```

*tools/docs_to_unix.py*

## 1.  解决：_pickle.UnpicklingError: the STRING opcode argument must be quoted

出现这个问题的原因是，pickle.load的文件是属于windows 下的docs 需要将其转化成Unix 下的文件。

“pickle文件必须使用Unix新行，否则至少Python 3.4的C pickle解析器会失败，例外：pickle.UnpicklingError：必须引用STRING操作码参数我认为某些git版本可能正在改变Unix新行（ '\ n'）到DOS行（'\ r \ n'）。

您可以使用此代码将“word_data.pkl”更改为“word_data_unix.pkl”，然后在脚本“nb_author_id.py”上使用新的.pkl文件：word_data_unix.pkl

``` 
#!/usr/bin/env python
"""
convert dos linefeeds (crlf) to unix (lf)
usage: dos2unix.py 
"""
original = "word_data.pkl"
destination = "word_data_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()
with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

print("Done. Saved %s bytes." % (len(content)-outsize))
```



原文：https://blog.csdn.net/weixin_37206602/article/details/89382222 


## 2.  解决python3的pickle.load错误：a bytes-like object is required, not 'str'

最近在python3下使用pickle.load时出现了错误。

错误信息如下：
    data_dict = pickle.load(data_file)

TypeError: a bytes-like object is required, not 'str'

经过几番查找，发现是Python3和Python2的字符串兼容问题，因为数据文件是在Python2下序列化的，所以使用Python3读取时，需要将‘str’转化为'bytes'。

```
import pickle

class StrToBytes:
    def __init__(self, fileobj):
        self.fileobj = fileobj
    def read(self, size):
        return self.fileobj.read(size).encode()
    def readline(self, size=-1):
        return self.fileobj.readline(size).encode()

with open('word_data.pkl', 'r') as data_file:
    data_dict = pickle.load(StrToBytes(data_file))
 
with open('word_data_fix.pkl', 'wb') as write_file:
     pickle.dump(data_dict, write_file)
```
经过这样一个转化后，就可以正确读取数据了。

原文：https://blog.csdn.net/junlee87/article/details/78780831 



## 3. TypeError: translate() takes exactly one argument (2 given)

*tools/parse_out_email_text.py*

```
file \ud120-projects\final_project\tester.py, line 27
  def test_classifier(clf, dataset, feature_list, folds=1000):
E       fixture 'clf' not found
>       available fixtures: cache, capfd, capfdbinary, caplog, capsys, capsysbinary, doctest_namespace, monkeypatch, pytestconfig, record_property, record_testsuite_property, record_xml_attribute, recwarn, tmp_path, tmp_path_factory, tmpdir, tmpdir_factory
>       use 'pytest --fixtures [testpath]' for help on them.
```

```
pip install pytest-sanic
pip install sanic
```
安装sanic时出错
```
error: Microsoft Visual C++ 14.0 is required. Get it with "Microsoft 

Visual C++ Build Tools": https://visualstudio.microsoft.com/downloads/
```
到这个地址下载，安装[Visual C++ Build Tools](<http://www.xdowns.com/soft/38/138/2017/Soft_226169.html>)
![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.2izf5x8nj4k.png)

## 4. “TypeError: 'StratifiedShuffleSplit' object is not iterable”

*final_project/tester.py*

[stackoverflow](<https://stackoverflow.com/questions/53899066/what-could-be-the-reason-for-typeerror-stratifiedshufflesplit-object-is-not>)

```
from sklearn.model_selection import StratifiedShuffleSplit
[..]
#cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
cv = StratifiedShuffleSplit(n_splits=folds, random_state=42)
[..]
#for train_idx, test_idx in cv:
for train_idx, test_idx in cv.split(features, labels):
[..]
```

