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


# 将相似度结果写入文件
def save_result(result_path, similarity_score):
    try:
        with open(result_path, 'w', encoding="utf-8") as result_file:
            result_file.write(f"文章相似度：{similarity_score:.2f}")
    except Exception as e:
        print(f"写入文件 {result_path} 时出错：{e}")
    return


# 主函数
def main():
    try:
        # 获取用户输入的文件路径
        # original_file_path = input("请输入原文文件的路径：").strip()
        # plagiarized_file_path = input("请输入抄袭版文件的路径：").strip()
        # result_file_path = input("请输入保存相似度结果的文件路径：").strip()

        # original_file_path = 'C:/Users/zj/Desktop/czoj/text/orig.txt'
        # plagiarized_file_path = 'C:/Users/zj/Desktop/czoj/text/orig_0.8_add.txt'
        # result_file_path = 'C:/Users/zj/Desktop/czoj/text/answer.txt'

        parser = argparse.ArgumentParser(description="文章相似度计算工具")
        parser.add_argument('original_file_path', type=str, help="原文文件的路径")
        parser.add_argument('plagiarized_file_path', type=str, help="抄袭版文件的路径")
        parser.add_argument('result_file_path', type=str, help="保存相似度结果的文件路径")

        args = parser.parse_args()

        # 获取用户输入的文件路径并赋值给原有变量
        original_file_path = args.original_file_path
        plagiarized_file_path = args.plagiarized_file_path
        result_file_path = args.result_file_path

        # 检查文件是否存在
        if not os.path.exists(original_file_path) or not os.path.exists(plagiarized_file_path):
            raise FileNotFoundError("输入的文件路径无效。请检查路径是否正确。")

        # 读取文件内容
        original_text = read_file(original_file_path)
        plagiarized_text = read_file(plagiarized_file_path)

        if original_text is None or plagiarized_text is None:
            raise ValueError("文件内容读取失败，无法继续。")

        # 文本预处理
        tokens1 = preprocess_text(original_text)
        tokens2 = preprocess_text(plagiarized_text)

        # 计算相似度
        similarity_score = calculate_cosine_similarity(tokens1, tokens2)

        if similarity_score is not None:
            print(f"文章相似度：{similarity_score:.2f}")
            save_result(result_file_path, similarity_score)
        else:
            print("相似度计算失败。")
        return
    except Exception as e:
        print(f"程序运行时出现错误：{e}")
        raise


if __name__ == '__main__':
    main()
    # 性能分析代码
    lp = LineProfiler()
    lp.add_function(main)
    lp.add_function(read_file)
    lp.add_function(preprocess_text)
    lp.add_function(calculate_cosine_similarity)
    lp.add_function(save_result)
    test_func = lp(main)
    test_func()
    lp.print_stats()


