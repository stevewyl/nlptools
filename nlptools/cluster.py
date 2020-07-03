import os
import re
import unicodedata
from collections import defaultdict

import jieba
import numpy as np
import scipy.spatial
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from tqdm import tqdm

from nlptools.preprocess import clean_text


class Cluster:
    def __init__(self, device="cpu"):
        self.encoder = SentenceTransformer('distiluse-base-multilingual-cased', device=device)
        self.stopwords = set(word.strip() for word in open("data/stopwords.txt"))

    def regex_clean(self, text):
        text = unicodedata.normalize("NFKC", text)
        text = clean_text(text)
        return text.strip()

    def pre_process(self, corpus):
        corpus = [self.regex_clean(doc) for doc in corpus]
        corpus = sorted(set(corpus))
        return corpus

    def cosine_sim(self, vec_a, vec_b):
        return 1 - scipy.spatial.distance.cdist(vec_a, vec_b, "cosine")

    def encode(self, corpus):
        return np.stack(self.encoder.encode(corpus), axis=0)

    def segment(self, text, remove_stop=True):
        words = list(jieba.cut(text))
        if remove_stop:
            words = list(filter(lambda x: x not in self.stopwords, words))
            words = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s]", "", " ".join(words))
            words = re.sub(r"\s{2,}", " ", words.strip()).split(" ")
        return words

    def cluster(self):
        raise NotImplementedError()

    def fit_transform(self):
        raise NotImplementedError()

    def load_vector(self, corpus, vec_fn):
        vecs = []
        with open(vec_fn) as fin:
            for idx, line in enumerate(fin):
                doc, vec = line.strip().split("\t", 1)
                assert doc == corpus[idx]
                vec = list(map(float, vec.split(" ")))
                vecs.append(vec)
        return np.stack(vecs, 0)

    def save_vector(self, corpus, vectors, vec_fn):
        with open(vec_fn, "w") as fw:
            for doc, vec in zip(corpus, vectors):
                vec = " ".join(list(map(str, [round(v, 4) for v in vec])))
                fw.write(f"{doc}\t{vec}\n")

    def show_clusters(self):
        raise NotImplementedError()


class KMeansCluster(Cluster):
    def __init__(self):
        """
        KMeans文本聚类
        """
        super(KMeansCluster, self).__init__()

    def cluster(self, vecs, num_clusters):
        model = KMeans(num_clusters)
        model.fit(vecs)
        return model

    def fit_transform(self, corpus, num_clusters, vec_fn=""):
        corpus = self.pre_process(corpus)
        if os.path.exists(vec_fn):
            vectors = self.load_vector(corpus, vec_fn)
            print("vector loaded")
        else:
            vectors = self.encode(corpus)
            if vec_fn:
                self.save_vector(corpus, vectors, vec_fn)
                print("vector saved")
        cluster_model = self.cluster(vectors, num_clusters)
        center_vectors = cluster_model.cluster_centers_
        cluster_assignment = cluster_model.labels_
        clustered_doc = defaultdict(list)
        for sentence_id, cluster_id in enumerate(cluster_assignment):
            clustered_doc[int(cluster_id)].append(corpus[sentence_id])
        clustered_doc = sorted(clustered_doc.items(), key=lambda x: -len(x[1]))
        return center_vectors, clustered_doc


class SinglePassCluster(Cluster):
    def __init__(self):
        """
        Single-Pass是一种增量聚类算法，不需要指定类目数量
        通过设定相似度阈值来限定聚类数量
        """
        super(SinglePassCluster, self).__init__()
        self.word2cluster = defaultdict(list)

    def get_max_similarity(self, clustered_vec, vec):
        max_value = 0
        max_index = -1
        for k, cluster_vecs in clustered_vec.items():
            vec = vec.reshape(1, -1)
            cluster_vecs = np.stack(cluster_vecs, 0)
            try:
                sim_score = np.mean(self.cosine_sim(vec, cluster_vecs))
            except:
                print(vec.shape, cluster_vecs.shape)
                sim_score = 0
            if sim_score > max_value:
                max_value = sim_score
                max_index = k
            if sim_score > 0.95:
                break
        return max_index, max_value

    def get_cand_clusters(self, words):
        cand_cluster_ids = []
        for word in words:
            cand_cluster_ids.extend(self.word2cluster.get(word, []))
        return list(set(cand_cluster_ids))

    def cluster(self, corpus, vecs, theta):
        num_topic = 0
        clustered_vec = defaultdict(list)
        clustered_doc = defaultdict(list)
        for vec, doc in tqdm(zip(vecs, corpus)):
            words = self.segment(doc)
            if num_topic == 0:
                cid = num_topic
                clustered_vec[cid].append(vec)
                clustered_doc[cid].append(doc)
                num_topic += 1
            else:
                cand_cluster_ids = self.get_cand_clusters(words)
                if cand_cluster_ids:
                    cand_vecs = {cid: clustered_vec[cid] for cid in cand_cluster_ids}
                else:
                    cand_vecs = clustered_vec
                    cand_cluster_ids = [i for i in range(len(cand_vecs))]
                max_index, max_value = self.get_max_similarity(cand_vecs, vec)
                if max_value >= theta and max_index != -1:
                    cid = max_index
                    clustered_vec[cid].append(vec)
                    clustered_doc[cid].append(doc)
                else:
                    cid = num_topic
                    clustered_vec[cid].append(vec)
                    clustered_doc[cid].append(doc)
                    num_topic += 1
            for word in list(set(words)):
                self.word2cluster[word].append(cid)
        center_vectors = {tid: np.mean(vecs) for tid, vecs in clustered_vec.items()}
        return center_vectors, clustered_doc

    def fit_transform(self, corpus, theta=0.5, vec_fn=""):
        corpus = self.pre_process(corpus)
        print(f"# of questions after preprocessing: {len(corpus)}")
        if os.path.exists(vec_fn):
            vectors = self.load_vector(corpus, vec_fn)
            print("vector loaded")
        else:
            vectors = self.encode(corpus)
            if vec_fn:
                self.save_vector(corpus, vectors, vec_fn)
                print("vector saved")
        print("vector encode done")
        center_vectors, clustered_doc = self.cluster(corpus, vectors, theta)
        clustered_doc = sorted(clustered_doc.items(), key=lambda x: -len(x[1]))
        return center_vectors, clustered_doc
