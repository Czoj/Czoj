import jieba
import gensim
import re
import os
import argparse
from line_profiler import LineProfiler


# 读取文件内容
def read_file(file_path):
    try:
        with open(file_path, 'r', encoding='UTF-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到。请检查路径是否正确。")
        return None
    except Exception as e:
        print(f"读取文件 {file_path} 时出错：{e}")
        return None


# 文本处理：进行jieba分词，并过滤掉标点符号、转义符号等特殊字符
def preprocess_text(text):
    # 去除标点符号和特殊字符
    text = re.sub(r"[^\w\s]", "", text)

    # 使用 jieba 进行分词
    tokens = jieba.lcut(text)

    # 去除空的分词结果
    filtered_tokens = [token for token in tokens if token.strip()]

    return filtered_tokens

# 计算文本的余弦相似度
def calculate_cosine_similarity(tokens1, tokens2):
    try:
        # 创建词典
        dictionary = gensim.corpora.Dictionary([tokens1, tokens2])
        # 将词汇映射到向量空间
        corpus = [dictionary.doc2bow(tokens) for tokens in [tokens1, tokens2]]
        # 计算相似度
        similarity_index = gensim.similarities.Similarity('-Similarity-index', corpus, num_features=len(dictionary))
        cosine_similarity = similarity_index[corpus[0]][1]
        return cosine_similarity
    except Exception as e:
        print(f"计算相似度时出错：{e}")
        return None



