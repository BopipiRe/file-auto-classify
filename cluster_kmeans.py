import os
import sys

import jieba
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def resource_path(path):
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, path)
    else:
        return os.path.join(os.path.abspath("."), path)


class Embedder:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        print(f"正在加载语义模型: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        print("模型加载完成")

    def embed(self, documents, batch_size=32):
        if not self.model:
            self.load_model()

        print("正在计算文档语义向量...")
        embeddings = self.model.encode(
            documents, batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True
        )
        print(f"向量计算完成，维度: {embeddings.shape}")
        return embeddings


class KMeansCluster:
    @classmethod
    def find_optimal_kmeans_clusters(cls, embeddings, max_k=15, min_k=2):
        """
        使用轮廓系数找到最优K-Means聚类数

        Args:
            embeddings: 嵌入向量
            max_k: 最大聚类数
            min_k: 最小聚类数

        Returns:
            tuple: (最优聚类数, 评估分数列表)
        """
        if len(embeddings) < 2:
            return 1, []

        silhouette_scores = []
        calinski_scores = []
        k_range = range(min_k, min(max_k + 1, len(embeddings)))

        if len(k_range) < 2:
            return len(k_range), []

        print("正在寻找最优聚类数...")
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)

            # 计算轮廓系数
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            silhouette_scores.append(silhouette_avg)

            # 计算Calinski-Harabasz指数
            calinski_avg = calinski_harabasz_score(embeddings, cluster_labels)
            calinski_scores.append(calinski_avg)

            print(f"  K={k}: 轮廓系数={silhouette_avg:.3f}, CH指数={calinski_avg:.1f}")

        # 选择轮廓系数最高的K值
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"最优聚类数: {optimal_k}")

        return optimal_k, list(zip(k_range, silhouette_scores, calinski_scores))

    @classmethod
    def perform_kmeans_clustering(cls, embeddings, n_clusters=None):
        """
        执行K-Means聚类

        Args:
            embeddings: 文档向量
            n_clusters: 聚类数（None表示自动选择）

        Returns:
            tuple: (聚类标签, KMeans模型)
        """
        # 自动选择 k：优先使用肘部法则 (elbow_method)，若无法确定则使用轮廓系数法
        selected_k = n_clusters
        if n_clusters is None:
            try:
                candidate_k, inertias = cls.elbow_method(embeddings)
                if candidate_k is not None:
                    selected_k = candidate_k
                    print(f"肘部法则推荐 K={selected_k}")
                else:
                    # 回退：使用轮廓系数寻找最优 k
                    selected_k, _ = cls.find_optimal_kmeans_clusters(embeddings)
                    print(f"肘部法则未能确定肘点，使用轮廓系数法推荐 K={selected_k}")
            except Exception as e:
                print(f"自动选 K 时出错: {e}，回退到轮廓系数法")
                selected_k, _ = cls.find_optimal_kmeans_clusters(embeddings)

        print(f"执行K-Means聚类 (K={selected_k})...")
        kmeans = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        # 评估聚类效果
        if len(set(cluster_labels)) > 1:
            try:
                silhouette_avg = silhouette_score(embeddings, cluster_labels)
                calinski_avg = calinski_harabasz_score(embeddings, cluster_labels)
                print(f"聚类评估 - 轮廓系数: {silhouette_avg:.3f}, CH指数: {calinski_avg:.1f}")
            except Exception:
                pass

        return cluster_labels, kmeans

    @classmethod
    def _generate_descriptive_name(
        cls, embeddings, cluster_labels, documents, label, kmeans_model, top_k=3, max_name_length=20
    ):
        """
        自动生成聚类的描述性名称

        Args:
            embeddings: 文档向量
            cluster_labels: 聚类标签
            documents: 原始文档列表
            label: 当前聚类的标签
            kmeans_model: 训练好的KMeans模型
            top_k: 用于生成名称的代表性文档数量
            max_name_length: 生成名称的最大长度

        Returns:
            str: 生成的聚类名称
        """
        from sklearn.metrics.pairwise import euclidean_distances

        # 获取属于该聚类的文档索引
        cluster_indices = np.where(cluster_labels == label)[0]

        if len(cluster_indices) == 0:
            return f"聚类_{label}"

        # 使用KMeans模型训练后的实际聚类中心
        cluster_center = kmeans_model.cluster_centers_[label].reshape(1, -1)

        # 找到最接近中心的文档
        cluster_embeddings = embeddings[cluster_indices]
        distances = euclidean_distances(cluster_embeddings, cluster_center).flatten()
        top_indices = cluster_indices[np.argsort(distances)[:top_k]]

        # 获取代表性文档的文本
        representative_docs = [documents[i] for i in top_indices]

        # 使用TF-IDF提取关键词
        try:
            # 对每个代表性文档分别处理
            doc_words = []
            for doc in representative_docs:
                words = jieba.cut(doc)
                # 过滤停用词和单字
                words = [w for w in words if len(w) > 1]
                doc_words.append(" ".join(words))

            if doc_words:
                # 使用TF-IDF提取关键词
                vectorizer = TfidfVectorizer(max_features=10, token_pattern=r"\S+")
                tfidf_matrix = vectorizer.fit_transform(doc_words)
                feature_names = vectorizer.get_feature_names_out()

                # 计算每个词的TF-IDF总分
                scores = tfidf_matrix.sum(axis=0).A1
                word_scores = dict(zip(feature_names, scores))

                # 选择得分最高的词
                top_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:5]

                # 组合成名称
                if top_words:
                    keywords = [word for word, score in top_words if score > 0.1]
                    if keywords:
                        name = " ".join(keywords[:3])
                        # 限制长度
                        if len(name) > max_name_length:
                            name = name[:max_name_length] + "..."
                        return name

        except Exception as e:
            raise Exception(f"生成聚类名称时出错: {e}")

    @classmethod
    def extract_semantic_cluster_names(cls, embeddings, cluster_labels, documents, kmeans_model):
        """
        自动生成聚类名称（基于聚类内容提取关键词）

        Args:
            embeddings: 文档向量
            cluster_labels: 聚类标签
            documents: 原始文档
            kmeans_model: 训练好的KMeans模型

        Returns:
            dict: 自动生成的聚类名称
        """
        cluster_names = {}

        print("正在自动生成聚类名称...")
        for label in np.unique(cluster_labels):
            cluster_names[label] = cls._generate_descriptive_name(
                embeddings, cluster_labels, documents, label, kmeans_model
            )
            print(f"  聚类 {label}: {cluster_names[label]}")

        return cluster_names

    @classmethod
    def elbow_method(cls, embeddings, min_k=2, max_k=15):
        """
        使用肘部法则（Elbow Method）评估不同 K 值的 inertia（SSE），并基于点到连接首尾两点直线的最大距离法
        给出候选的 K 值。

        Args:
            embeddings: 嵌入向量数组，形状 (n_samples, n_features)
            min_k: 最小聚类数（包含）
            max_k: 最大聚类数（包含）

        Returns:
            tuple: (candidate_k, inertia_list)
                - candidate_k: 推荐的聚类数（int）或 None（若无法判断）
                - inertia_list: 包含 (k, inertia) 的列表，方便绘图或进一步分析

        说明:
            - inertia 等同于 KMeans 的 SSE（簇内平方和），通常随 K 增大而下降。
            - 该方法使用“最大垂直距离到首尾连线”来自动定位肘点（常用启发式方法）。
        """
        try:
            n_samples = len(embeddings)
        except Exception:
            raise ValueError("embeddings 必须是可计算长度的数组或列表")

        if n_samples < 2:
            return None, []

        inertias = []
        k_list = list(range(min_k, min(max_k + 1, n_samples)))

        print("计算不同 K 的 inertia（SSE）...")
        for k in k_list:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(embeddings)
            inertias.append(float(kmeans.inertia_))
            print(f"  K={k}: inertia={kmeans.inertia_:.3f}")

        # 如果点太少，无法用肘部法则判断
        candidate_k = None
        if len(k_list) >= 3:
            ks = np.array(k_list)
            vals = np.array(inertias)

            # 直线首尾点
            p1 = np.array([ks[0], vals[0]])
            p2 = np.array([ks[-1], vals[-1]])
            line_vec = p2 - p1
            line_len = np.linalg.norm(line_vec)

            if line_len > 0:
                pts = np.vstack([ks, vals]).T
                vecs = pts - p1
                # 计算点到直线的垂直距离（2D cross product magnitude / line length）
                cross = np.abs(vecs[:, 0] * line_vec[1] - vecs[:, 1] * line_vec[0])
                dists = cross / line_len

                # 不考虑首尾端点（因为端点距离为 0）
                if len(dists) > 2:
                    interior_idx = np.arange(1, len(dists) - 1)
                    best_idx = interior_idx[np.argmax(dists[interior_idx])]
                    candidate_k = int(ks[best_idx])

        return candidate_k, list(zip(k_list, inertias))
