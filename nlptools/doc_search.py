import numpy as np
from gensim.models import KeyedVectors

from nlptools.metrics.distance import cosine_dist


class DocSearcher:
    def __init__(self, tokenizer, vec_fn, n_planes=10, n_universes=25):
        np.random.seed(0)

        self.embeddings = self.load_vec(vec_fn)
        self.n_planes =  n_planes
        self.n_universes = n_universes
        self.n_dims = self.embeddings.shape[1]
        self.planes_l = [np.random.normal(size=(self.n_dims, self.n_planes))
                         for _ in range(self.n_universes)]
        self.tokenizer = tokenizer

    def load_vec(self, vec_fn, binary=False):
        if vec_fn.endswith("bin"):
            binary = True
        return  KeyedVectors.load_word2vec_format(vec_fn, binary=binary)

    def get_doc_embedding(self, doc):
        doc_embedding = np.zeros(self.n_dims)
        for word in self.tokenizer.tokenize(doc):
            doc_embedding = np.add(doc_embedding, self.embeddings[word])
        return doc_embedding

    def get_doc_vecs(self, docs):
        id2doc_embed = {}
        doc_vec_l = []
        for i, doc in enumerate(docs):
            doc_embedding = self.get_doc_embedding(doc)
            id2doc_embed[i] = doc_embedding
            doc_vec_l.append(doc_embedding)
        doc_vec_matrix = np.vstack(doc_vec_l)
        return doc_vec_matrix, id2doc_embed

    def nearest_neighbor(self, v, candidates, k=1):
        sim_l = []
        for row in candidates:
            cos_sim = 1 - cosine_dist(row, v)
            sim_l.append(cos_sim)
        sorted_ids = np.argsort(sim_l)
        k_idx = sorted_ids[-k:]
        return k_idx

    def hash_value_of_vector(self, v, planes):
        dot_product = np.dot(v, planes)
        sign_of_dot_product = np.squeeze(np.sign(dot_product))
        h = [1 if sign > 0 else 0 for sign in sign_of_dot_product]
        h = np.squeeze(h)
        hash_value = 0
        for i in range(self.n_planes):
            hash_value += 2 ** i * h[i]
        return int(hash_value)

    def make_hash_table(self, planes):
        num_buckets = 2 ** self.n_planes
        hash_table = {i: [] for i in range(num_buckets)}
        id_table = {i: [] for i in range(num_buckets)}
        for i, v in enumerate(self.doc_vec_matrix):
            h = self.hash_value_of_vector(v, planes)
            hash_table[h].append(v)
            id_table[h].append(i)
        return hash_table, id_table

    def build(self, docs):
        self.doc_vec_matrix, self.id2doc_embed = self.get_doc_vecs(docs)
        self.docs = docs
        self.hash_tables = []
        self.id_tables = []
        for universe_id in range(self.n_universes):
            print('working on hash universe #:', universe_id)
            planes = self.planes_l[universe_id]
            hash_table, id_table = self.make_hash_table(planes)
            self.hash_tables.append(hash_table)
            self.id_tables.append(id_table)

    def lookup_search(self, doc):
        doc_embedding = self.get_doc_embedding(doc)
        idx = np.argmax(1 - cosine_dist(self.doc_vec_matrix, doc_embedding))
        return self.docs[idx]

    def lsh_search(self, doc, k=1, num_unvervise_to_use=self.n_universes):
        assert num_unvervise_to_use <= self.n_universes
        vecs_candidates = []
        ids_candidates = []
        v = self.get_doc_embedding(doc)

        for universe_id in range(num_unvervise_to_use):
            planes = self.planes_l[universe_id]
            hash_value = self.hash_value_of_vector(v, planes)
            hash_table = self.hash_tables[universe_id]
            vecs_at_h = hash_table[hash_value]
            id_table = self.id_tables[universe_id]
            ids_at_h = id_table[hash_value]

            for i, new_id in enumerate(ids_at_h):
                doc_vec_i = vecs_at_h[i]
                vecs_candidates.append(doc_vec_i)
                ids_candidates.append(new_id)

        vec_candidates_arr = np.array(vecs_candidates)
        nearest_neighbor_l = nearest_neighbor(v, vec_candidates_arr, k)
        nearest_neighbor_id = [id_candidates[idx] for idx in nearest_neighbor_l]
        return nearest_neighbor_id

if __name__ == "__main__":
    doc_searcher = DocSearcher()