import unittest
import jieba
import os
import re
from main import read_file, preprocess_text, calculate_cosine_similarity, save_result

class TestSimilarityFunctions(unittest.TestCase):

    def setUp(self):
        # 创建两个测试文本
        self.test_text1 = "这是一个用于测试的文本。"
        self.test_text2 = "这个文本和第一个文本有一些相似之处。"
        self.empty_text = ""
        self.similarity_score_path = "test_similarity_score.txt"

        # 创建测试文件
        with open("test_file1.txt", "w", encoding="utf-8") as f1:
            f1.write(self.test_text1)
        with open("test_file2.txt", "w", encoding="utf-8") as f2:
            f2.write(self.test_text2)
        with open("empty_file.txt", "w", encoding="utf-8") as f3:
            f3.write(self.empty_text)

    def tearDown(self):
        # 删除测试文件
        os.remove("test_file1.txt")
        os.remove("test_file2.txt")
        os.remove("empty_file.txt")
        if os.path.exists(self.similarity_score_path):
            os.remove(self.similarity_score_path)

    def test_read_file(self):
        # 测试读取文件功能
        content1 = read_file("test_file1.txt")
        content2 = read_file("test_file2.txt")
        self.assertEqual(content1, self.test_text1)
        self.assertEqual(content2, self.test_text2)

        # 测试读取空文件
        content = read_file("empty_file.txt")
        self.assertEqual(content, self.empty_text)

        # 测试读取不存在的文件
        content = read_file("nonexistent_file.txt")
        self.assertIsNone(content)

    def test_preprocess_text(self):
        # 测试文本预处理功能
        tokens1 = preprocess_text(self.test_text1)
        tokens2 = preprocess_text(self.test_text2)

        # 输出实际分词结果以调试
        print("分词结果1:", tokens1)
        print("分词结果2:", tokens2)

        # 手动检查jieba分词的实际结果
        test_text1_cleaned = re.sub(r"[^\w\s]", "", self.test_text1)
        test_text2_cleaned = re.sub(r"[^\w\s]", "", self.test_text2)
        expected_tokens1 = jieba.lcut(test_text1_cleaned)
        expected_tokens2 = jieba.lcut(test_text2_cleaned)

        # 断言实际预处理结果与分词结果一致
        self.assertEqual(tokens1, expected_tokens1)
        self.assertEqual(tokens2, expected_tokens2)

        # 测试空文本
        empty_tokens = preprocess_text(self.empty_text)
        self.assertEqual(empty_tokens, [])

    def test_calculate_cosine_similarity(self):
        # 测试余弦相似度计算
        tokens1 = preprocess_text(self.test_text1)
        tokens2 = preprocess_text(self.test_text2)
        similarity_score = calculate_cosine_similarity(tokens1, tokens2)

        # 验证相似度是否为合理范围（0到1之间）
        self.assertGreaterEqual(similarity_score, 0.0)
        self.assertLessEqual(similarity_score, 1.0)

        # 测试两个完全相同的文本
        identical_tokens = preprocess_text(self.test_text1)
        similarity_score_identical = calculate_cosine_similarity(identical_tokens, identical_tokens)
        self.assertAlmostEqual(similarity_score_identical, 1.0, places=4)

        # 测试空文本和非空文本的相似度
        empty_tokens = preprocess_text(self.empty_text)
        similarity_score_with_empty = calculate_cosine_similarity(empty_tokens, tokens1)
        self.assertEqual(similarity_score_with_empty, 0.0)

    def test_save_result(self):
        # 测试保存相似度结果到文件
        similarity_score = 0.85
        save_result(self.similarity_score_path, similarity_score)

        with open(self.similarity_score_path, 'r', encoding="utf-8") as f:
            content = f.read()
        self.assertEqual(content, "文章相似度：0.85")

        # 测试保存到无权限文件路径（模拟异常情况）
        try:
            save_result("/invalid_path/test_similarity_score.txt", similarity_score)
        except Exception as e:
            self.assertIsInstance(e, Exception)

if __name__ == '__main__':
    unittest.main()
