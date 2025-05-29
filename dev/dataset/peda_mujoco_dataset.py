import h5py
import numpy as np
import torch
import pickle
from time import time

from cleandiffuser.dataset.base_dataset import BaseDataset
from cleandiffuser.dataset.dataset_utils import GaussianNormalizer, dict_apply, MinMaxNormalizer
from dev.utils.utils import TrajSparseTable, LRUDatasetCache

def check_dominated(obj_batch, obj, tolerance=0):
    return (np.logical_and((obj_batch * (1-tolerance) >= obj).all(axis=1), (obj_batch * (1-tolerance) > obj).any(axis=1))).any()

# return sorted indices of nondominated objs
def undominated_indices(obj_batch_input, tolerance=0):
    obj_batch = np.array(obj_batch_input)
    sorted_indices = np.argsort(obj_batch.T[0])
    indices = []
    for idx in sorted_indices:
        if (obj_batch[idx] >= 0).all() and not check_dominated(obj_batch, obj_batch[idx], tolerance):
            indices.append(idx)
    return indices

class PEDAMuJoCoDataset(BaseDataset):
    def __init__(
            self,
            dataset: str="MO-Hopper-v2_50000_amateur_uniform",
            terminal_penalty=100,
            horizon=32,
            max_path_length=500,
            avg=True,
            gamma=0,
            normalize_rewards=True,
            eps=1e-3,
            discount=0.99,
            force_override=False,
            read_only=False,
            weighted_score=False
    ):
        super().__init__()

        from time import time
        
        with open("dev/data/cache/LRUDatasetCache.pkl", "rb") as f:
            cache = pickle.load(file=f)

        need_process = True
        if not force_override and cache.exists((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score)):
            print("Loading cached dataset...")
            try:
                cached_dataset = cache.get((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score))
                self.__dict__.update(cached_dataset)
                need_process = False
            except:
                print("Failed to load cached dataset, reprocessing...")

        if need_process:
            env, _, _, _ = dataset.split("_")

            print(f"Processing dataset {dataset}{', This may take a while.' if (normalize_rewards and eps > 0) else '...'}")

            trajectories = []
            with open(f'./dev/data/{env}/{dataset}.pkl', "rb") as f:
                trajectories.extend(pickle.load(f))
            
            self.o_dim = trajectories[0]["observations"].shape[-1]
            self.a_dim = trajectories[0]["actions"].shape[-1]
            self.pref_dim = trajectories[0]["preference"].shape[-1]
            self.n_trajs = len(trajectories)

            self.horizon = horizon
            self.discount = discount ** np.arange(max_path_length, dtype=np.float32)
            self.discount = self.discount.reshape((-1, 1)).repeat(self.pref_dim, axis=-1)

            if avg:
                self.avg = np.arange(max_path_length, 0, -1, dtype=np.float32)
                self.avg = self.avg.reshape((-1, 1)).repeat(self.pref_dim, axis=-1)

            self.seq_obs = np.zeros((self.n_trajs, max_path_length, self.o_dim), dtype=np.float32)
            self.seq_act = np.zeros((self.n_trajs, max_path_length, self.a_dim), dtype=np.float32)
            self.seq_rew = np.zeros((self.n_trajs, max_path_length, self.pref_dim), dtype=np.float32)
            self.seq_pref = np.zeros((self.n_trajs, self.pref_dim), dtype=np.float32)
            self.seq_score = np.zeros((self.n_trajs, max_path_length, self.pref_dim), dtype=np.float32)
            seq_len = np.zeros((self.n_trajs,), dtype=np.int32)

            for i, traj in enumerate(trajectories):
                length = traj["observations"].shape[0]
                self.seq_obs[i, :length] = traj["observations"]
                self.seq_act[i, :length] = traj["actions"]
                self.seq_pref[i] = traj["preference"][0]
                self.seq_rew[i, :length] = traj["raw_rewards"]
                self.seq_score[i, :length] = traj["raw_rewards"]
                seq_len[i] = length

                if length != 500 and traj["terminals"][length-1]:
                    self.seq_score[i, length-1] -= terminal_penalty

                if gamma == 1:
                    self.seq_score[i, :length] = np.cumsum(self.seq_score[i, :length][::-1], axis=0)[::-1]
                elif gamma > 0:
                    for j in range(length-2, -1, -1):
                        self.seq_score[i, j] += gamma * self.seq_score[i, j+1]
                
                if avg:
                    self.seq_score[i] /= self.avg

                # max_start = length - horizon
                # self.indices += [(i, start, start + horizon) for start in range(0, max_start + 1, 1)]
            flatten_obs = self.seq_obs.reshape(-1, self.o_dim)
            flatten_act = self.seq_act.reshape(-1, self.a_dim)

            self.normalizers = {
                "state": GaussianNormalizer(flatten_obs),
                "action": MinMaxNormalizer(flatten_act)
            }
            # [:] result in a view rather than copy
            flatten_obs[:] = self.normalizers["state"].normalize(flatten_obs)
            flatten_act[:] = self.normalizers["action"].normalize(flatten_act)

            sorted_index = np.argsort(self.seq_pref[:,0])
            self.seq_obs = self.seq_obs[sorted_index]
            self.seq_act = self.seq_act[sorted_index]
            self.seq_rew = self.seq_rew[sorted_index]
            self.seq_pref = self.seq_pref[sorted_index]
            self.seq_score = self.seq_score[sorted_index]
            seq_len = seq_len[sorted_index]

            self.indices = []
            for i in range(self.n_trajs):
                length = seq_len[i]
                max_start = length - horizon
                self.indices += [(i, start, start + horizon) for start in range(0, max_start + 1, 1)]
            
            self.seq_front = np.zeros((self.n_trajs,), dtype=np.bool8)
            self.sum_seq_rew = self.seq_rew.sum(1)

            self.front_idx = undominated_indices(self.sum_seq_rew)
            self.seq_front[self.front_idx] = True
            
            self.front_rew = self.sum_seq_rew[self.front_idx]
            self.front_pref = self.seq_pref[self.front_idx]

            if normalize_rewards:
                st = TrajSparseTable(self.seq_score)

                if eps > 0:
                    # neighbor normalizing
                    que = []
                    r = 0
                    normalized_seq_score = np.zeros_like(self.seq_score)

                    for i in range(self.n_trajs):
                        # slide window
                        while que and self.seq_pref[que[0], 0] < self.seq_pref[i, 0] - eps:
                            que.pop(0)
                        while r < self.n_trajs and self.seq_pref[r, 0] <= self.seq_pref[i, 0] + eps:
                            que.append(r)
                            r += 1
                        
                        # sparse table query, faster
                        neighbor_max_score = st.max(que[0], que[-1])
                        neighbor_min_score = st.min(que[0], que[-1])

                        # brute force
                        # neighbor_max_score = np.max(self.seq_score[que], axis=0)
                        # neighbor_min_score = np.min(self.seq_score[que], axis=0)

                        normalized_seq_score[i] = (self.seq_score[i] - neighbor_min_score) / (neighbor_max_score - neighbor_min_score + 1e-6)
                    self.seq_score = normalized_seq_score

                else:
                    max_score = np.max(self.seq_score, axis=0)
                    min_score = np.min(self.seq_score, axis=0)
                    self.seq_score = (self.seq_score - min_score) / (max_score - min_score + 1e-6)
            
            if weighted_score:
                self.seq_score = self.seq_score * self.seq_pref[:, None, :]
        
        if not read_only:
            with open("dev/data/cache/LRUDatasetCache.pkl", "rb") as f:
                cache = pickle.load(file=f)
            
            cache.put((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score), self.__dict__)

            with open("dev/data/cache/LRUDatasetCache.pkl", "wb") as f:
                pickle.dump(cache, file=f)
        print("Done.")

    def get_normalizer(self):
        return self.normalizers
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        scores = self.seq_score[path_idx, start:end]
        values = (scores * self.discount[:scores.shape[0]]).sum(0)

        data = {
            'obs': {
            'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'val': values,
            'pref': self.seq_pref[path_idx],
            'score': self.seq_score[path_idx, start:end]}
        
        # TODO: normalize val by horizon

        return dict_apply(data, torch.tensor)

class ShatteredPEDAMuJoCoDataset(BaseDataset):
    def __init__(
            self,
            dataset: str="MO-Hopper-v2_50000_amateur_uniform",
            terminal_penalty=100,
            horizon=32,
            max_path_length=500,
            avg=True,
            gamma=0,
            normalize_rewards=True,
            eps=1e-3,
            discount=0.99,
            force_override=False,
            read_only=False,
            weighted_score=False,
            dropout_percent=0.3,
            dropout_points=3,
            side=False,
    ):
        super().__init__()

        from time import time
        
        with open("dev/data/cache/LRUDatasetCache.pkl", "rb") as f:
            cache = pickle.load(file=f)

        need_process = True
        if not force_override and cache.exists((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score, dropout_percent, dropout_points, side)):
            print("Loading cached dataset...")
            try:
                cached_dataset = cache.get((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score, dropout_percent, dropout_points, side))
                self.__dict__.update(cached_dataset)
                need_process = False
            except:
                print("Failed to load cached dataset, reprocessing...")

        if need_process:
            env, _, _, _ = dataset.split("_")

            print(f"Processing dataset {dataset}{', This may take a while.' if (normalize_rewards and eps > 0) else '...'}")

            trajectories = []
            with open(f'./dev/data/{env}/{dataset}.pkl', "rb") as f:
                trajectories.extend(pickle.load(f))
            
            self.o_dim = trajectories[0]["observations"].shape[-1]
            self.a_dim = trajectories[0]["actions"].shape[-1]
            self.pref_dim = trajectories[0]["preference"].shape[-1]
            self.n_trajs = len(trajectories)

            self.horizon = horizon
            self.discount = discount ** np.arange(max_path_length, dtype=np.float32)
            self.discount = self.discount.reshape((-1, 1)).repeat(self.pref_dim, axis=-1)

            if avg:
                self.avg = np.arange(max_path_length, 0, -1, dtype=np.float32)
                self.avg = self.avg.reshape((-1, 1)).repeat(self.pref_dim, axis=-1)

            self.seq_obs = np.zeros((self.n_trajs, max_path_length, self.o_dim), dtype=np.float32)
            self.seq_act = np.zeros((self.n_trajs, max_path_length, self.a_dim), dtype=np.float32)
            self.seq_rew = np.zeros((self.n_trajs, max_path_length, self.pref_dim), dtype=np.float32)
            self.seq_pref = np.zeros((self.n_trajs, self.pref_dim), dtype=np.float32)
            self.seq_score = np.zeros((self.n_trajs, max_path_length, self.pref_dim), dtype=np.float32)
            self.seq_front = np.ones((self.n_trajs,), dtype=np.bool8)
            seq_len = np.zeros((self.n_trajs,), dtype=np.int32)

            for i, traj in enumerate(trajectories):
                length = traj["observations"].shape[0]
                self.seq_obs[i, :length] = traj["observations"]
                self.seq_act[i, :length] = traj["actions"]
                self.seq_pref[i] = traj["preference"][0]
                self.seq_rew[i, :length] = traj["raw_rewards"]
                self.seq_score[i, :length] = traj["raw_rewards"]
                seq_len[i] = length

                if traj["terminals"][length-1]:
                    self.seq_score[i, length-1] -= terminal_penalty

                if gamma == 1:
                    self.seq_score[i, :length] = np.cumsum(traj["raw_rewards"][::-1], axis=0)[::-1]
                elif gamma > 0:
                    for j in range(length-2, -1, -1):
                        self.seq_score[i, j] += gamma * self.seq_score[i, j+1]
                
                if avg:
                    self.seq_score[i] /= self.avg

                # max_start = length - horizon
                # self.indices += [(i, start, start + horizon) for start in range(0, max_start + 1, 1)]
            flatten_obs = self.seq_obs.reshape(-1, self.o_dim)
            self.normalizers = {
                "state": GaussianNormalizer(flatten_obs)
            }
            # [:] result in a view rather than copy
            flatten_obs[:] = self.normalizers["state"].normalize(flatten_obs)

            sorted_index = np.argsort(self.seq_pref[:,0])
            self.seq_obs = self.seq_obs[sorted_index]
            self.seq_act = self.seq_act[sorted_index]
            self.seq_rew = self.seq_rew[sorted_index]
            self.seq_pref = self.seq_pref[sorted_index]
            self.seq_score = self.seq_score[sorted_index]
            seq_len = seq_len[sorted_index]

            if side:
                dropout_radius = int(self.n_trajs * dropout_percent / 2)
                dropout_centres = np.linspace(0, self.n_trajs-1, 2, dtype=np.int32)[1:-1]
                self.dropout_indices = [i for i in range(0, dropout_radius+1)] + [self.n_trajs-1-i for i in range(0, dropout_radius+1)]
            else:
                dropout_radius = int(self.n_trajs * dropout_percent / dropout_points / 2)
                dropout_centres = np.linspace(0, self.n_trajs-1, dropout_points+2, dtype=np.int32)[1:-1]
                self.dropout_indices = [i+j for i in dropout_centres for j in range(-dropout_radius, dropout_radius+1)]

            self.dropout_prefs = self.seq_pref[self.dropout_indices]
            self.seq_obs = np.delete(self.seq_obs, self.dropout_indices, axis=0)
            self.seq_act = np.delete(self.seq_act, self.dropout_indices, axis=0)
            self.seq_rew = np.delete(self.seq_rew, self.dropout_indices, axis=0)
            self.seq_pref = np.delete(self.seq_pref, self.dropout_indices, axis=0)
            self.seq_score = np.delete(self.seq_score, self.dropout_indices, axis=0)
            seq_len = np.delete(seq_len, self.dropout_indices, axis=0)

            self.n_trajs -= len(self.dropout_indices)

            self.indices = []
            for i in range(self.n_trajs):
                length = seq_len[i]
                max_start = length - horizon
                self.indices += [(i, start, start + horizon) for start in range(0, max_start + 1, 1)]

            self.seq_front = np.zeros((self.n_trajs,), dtype=np.bool8)
            sum_seq_rew = self.seq_rew.sum(1)

            self.front_idx = undominated_indices(sum_seq_rew)
            self.seq_front[self.front_idx] = True
            
            self.front_rew = sum_seq_rew[self.front_idx]
            self.front_pref = self.seq_pref[self.front_idx]

            if normalize_rewards:
                st = TrajSparseTable(self.seq_score)

                if eps > 0:
                    # neighbor normalizing
                    que = []
                    r = 0
                    normalized_seq_score = np.zeros_like(self.seq_score)

                    for i in range(self.n_trajs):
                        # slide window
                        while que and self.seq_pref[que[0], 0] < self.seq_pref[i, 0] - eps:
                            que.pop(0)
                        while r < self.n_trajs and self.seq_pref[r, 0] <= self.seq_pref[i, 0] + eps:
                            que.append(r)
                            r += 1
                        
                        # sparse table query, faster
                        neighbor_max_score = st.max(que[0], que[-1])
                        neighbor_min_score = st.min(que[0], que[-1])

                        # brute force
                        # neighbor_max_score = np.max(self.seq_score[que], axis=0)
                        # neighbor_min_score = np.min(self.seq_score[que], axis=0)

                        normalized_seq_score[i] = (self.seq_score[i] - neighbor_min_score) / (neighbor_max_score - neighbor_min_score + 1e-6)
                    self.seq_score = normalized_seq_score

                else:
                    max_score = np.max(self.seq_score, axis=0)
                    min_score = np.min(self.seq_score, axis=0)
                    self.seq_score = (self.seq_score - min_score) / (max_score - min_score + 1e-6)
            
            if weighted_score:
                self.seq_score = self.seq_score * self.seq_pref[:, None, :]

        if not read_only:
            with open("dev/data/cache/LRUDatasetCache.pkl", "rb") as f:
                cache = pickle.load(file=f)
            
            cache.put((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score, dropout_percent, dropout_points, side), self.__dict__)

            with open("dev/data/cache/LRUDatasetCache.pkl", "wb") as f:
                pickle.dump(cache, file=f)
        print("Done.")

    def get_normalizer(self):
        return self.normalizers["state"]
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        scores = self.seq_score[path_idx, start:end]
        values = (scores * self.discount[:scores.shape[0]]).sum(0)

        data = {
            'obs': {
            'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'val': values,
            'pref': self.seq_pref[path_idx],
            'score': self.seq_score[path_idx, start:end]}
        
        # TODO: normalize val by horizon

        return dict_apply(data, torch.tensor)

class PEDAMuJoCoDataset3Obj(BaseDataset):
    def __init__(
            self,
            dataset: str="MO-Hopper-v2_50000_amateur_uniform",
            terminal_penalty=100,
            horizon=32,
            max_path_length=500,
            avg=True,
            gamma=0,
            normalize_rewards=True,
            eps=3e-3,
            discount=0.99,
            force_override=False,
            read_only=False,
            weighted_score=False
    ):
        super().__init__()

        from time import time
        
        with open("dev/data/cache/LRUDatasetCache.pkl", "rb") as f:
            cache = pickle.load(file=f)

        need_process = True
        if not force_override and cache.exists((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score)):
            print("Loading cached dataset...")
            try:
                cached_dataset = cache.get((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score))
                self.__dict__.update(cached_dataset)
                need_process = False
            except:
                print("Failed to load cached dataset, reprocessing...")

        if need_process:
            env, _, _, _ = dataset.split("_")

            print(f"Processing dataset {dataset}{', This may take a while.' if (normalize_rewards and eps > 0) else '...'}")

            trajectories = []
            with open(f'./dev/data/{env}/{dataset}.pkl', "rb") as f:
                trajectories.extend(pickle.load(f))
            
            self.o_dim = trajectories[0]["observations"].shape[-1]
            self.a_dim = trajectories[0]["actions"].shape[-1]
            self.pref_dim = trajectories[0]["preference"].shape[-1]
            self.n_trajs = len(trajectories)

            self.horizon = horizon
            self.discount = discount ** np.arange(max_path_length, dtype=np.float32)
            self.discount = self.discount.reshape((-1, 1)).repeat(self.pref_dim, axis=-1)

            if avg:
                self.avg = np.arange(max_path_length, 0, -1, dtype=np.float32)
                self.avg = self.avg.reshape((-1, 1)).repeat(self.pref_dim, axis=-1)

            self.seq_obs = np.zeros((self.n_trajs, max_path_length, self.o_dim), dtype=np.float32)
            self.seq_act = np.zeros((self.n_trajs, max_path_length, self.a_dim), dtype=np.float32)
            self.seq_rew = np.zeros((self.n_trajs, max_path_length, self.pref_dim), dtype=np.float32)
            self.seq_pref = np.zeros((self.n_trajs, self.pref_dim), dtype=np.float32)
            self.seq_score = np.zeros((self.n_trajs, max_path_length, self.pref_dim), dtype=np.float32)
            seq_len = np.zeros((self.n_trajs,), dtype=np.int32)

            for i, traj in enumerate(trajectories):
                length = traj["observations"].shape[0]
                self.seq_obs[i, :length] = traj["observations"]
                self.seq_act[i, :length] = traj["actions"]
                self.seq_pref[i] = traj["preference"][0]
                self.seq_rew[i, :length] = traj["raw_rewards"]
                self.seq_score[i, :length] = traj["raw_rewards"]
                seq_len[i] = length

                if traj["terminals"][length-1]:
                    self.seq_score[i, length-1] -= terminal_penalty

                if gamma == 1:
                    self.seq_score[i, :length] = np.cumsum(traj["raw_rewards"][::-1], axis=0)[::-1]
                elif gamma > 0:
                    for j in range(length-2, -1, -1):
                        self.seq_score[i, j] += gamma * self.seq_score[i, j+1]
                
                if avg:
                    self.seq_score[i] /= self.avg

                # max_start = length - horizon
                # self.indices += [(i, start, start + horizon) for start in range(0, max_start + 1, 1)]
            flatten_obs = self.seq_obs.reshape(-1, self.o_dim)
            self.normalizers = {
                "state": GaussianNormalizer(flatten_obs)
            }
            # [:] result in a view rather than copy
            flatten_obs[:] = self.normalizers["state"].normalize(flatten_obs)

            sorted_index = np.argsort(self.seq_pref[:,0])
            self.seq_obs = self.seq_obs[sorted_index]
            self.seq_act = self.seq_act[sorted_index]
            self.seq_rew = self.seq_rew[sorted_index]
            self.seq_pref = self.seq_pref[sorted_index]
            self.seq_score = self.seq_score[sorted_index]
            seq_len = seq_len[sorted_index]

            self.indices = []
            for i in range(self.n_trajs):
                length = seq_len[i]
                max_start = length - horizon
                self.indices += [(i, start, start + horizon) for start in range(0, max_start + 1, 1)]
            
            self.seq_front = np.zeros((self.n_trajs,), dtype=np.bool8)
            sum_seq_rew = self.seq_rew.sum(1)

            self.front_idx = undominated_indices(sum_seq_rew)
            self.seq_front[self.front_idx] = True
            
            self.front_rew = sum_seq_rew[self.front_idx]
            self.front_pref = self.seq_pref[self.front_idx]

            if normalize_rewards:
                # st = TrajSparseTable(self.seq_score)

                if eps > 0:
                    # neighbor normalizing
                    r = 0
                    normalized_seq_score = np.zeros_like(self.seq_score)

                    for i in range(self.n_trajs):
                        tmp = self.seq_pref - self.seq_pref[i]
                        tmp = np.linalg.norm(tmp, 2, axis=1)
                        que = np.where(tmp < eps * np.sqrt(2))[0]

                        # brute force
                        neighbor_max_score = np.max(self.seq_score[que], axis=0)
                        neighbor_min_score = np.min(self.seq_score[que], axis=0)

                        normalized_seq_score[i] = (self.seq_score[i] - neighbor_min_score) / (neighbor_max_score - neighbor_min_score + 1e-6)
                    self.seq_score = normalized_seq_score

                else:
                    max_score = np.max(self.seq_score, axis=0)
                    min_score = np.min(self.seq_score, axis=0)
                    self.seq_score = (self.seq_score - min_score) / (max_score - min_score + 1e-6)
            
            if weighted_score:
                self.seq_score = self.seq_score * self.seq_pref[:, None, :]
        
        if not read_only:
            with open("dev/data/cache/LRUDatasetCache.pkl", "rb") as f:
                cache = pickle.load(file=f)
            
            cache.put((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score), self.__dict__)

            with open("dev/data/cache/LRUDatasetCache.pkl", "wb") as f:
                pickle.dump(cache, file=f)
        print("Done.")

    def get_normalizer(self):
        return self.normalizers["state"]
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        scores = self.seq_score[path_idx, start:end]
        values = (scores * self.discount[:scores.shape[0]]).sum(0)

        data = {
            'obs': {
            'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'val': values,
            'pref': self.seq_pref[path_idx],
            'score': self.seq_score[path_idx, start:end]}
        
        # TODO: normalize val by horizon

        return dict_apply(data, torch.tensor)

class ShatteredPEDAMuJoCoDataset3Obj(BaseDataset):
    def __init__(
            self,
            dataset: str="MO-Hopper-v2_50000_amateur_uniform",
            terminal_penalty=100,
            horizon=32,
            max_path_length=500,
            avg=True,
            gamma=0,
            normalize_rewards=True,
            eps=3e-3,
            discount=0.99,
            force_override=False,
            read_only=False,
            weighted_score=False,
            dropout_percent=0.3,
            dropout_points=3,
            side=False,
    ):
        super().__init__()

        from time import time
        
        with open("dev/data/cache/LRUDatasetCache.pkl", "rb") as f:
            cache = pickle.load(file=f)

        need_process = True
        if not force_override and cache.exists((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score, dropout_percent, dropout_points, side)):
            print("Loading cached dataset...")
            try:
                cached_dataset = cache.get((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score, dropout_percent, dropout_points, side))
                self.__dict__.update(cached_dataset)
                need_process = False
            except:
                print("Failed to load cached dataset, reprocessing...")

        if need_process:
            env, _, _, _ = dataset.split("_")

            print(f"Processing dataset {dataset}{', This may take a while.' if (normalize_rewards and eps > 0) else '...'}")

            trajectories = []
            with open(f'./dev/data/{env}/{dataset}.pkl', "rb") as f:
                trajectories.extend(pickle.load(f))
            
            self.o_dim = trajectories[0]["observations"].shape[-1]
            self.a_dim = trajectories[0]["actions"].shape[-1]
            self.pref_dim = trajectories[0]["preference"].shape[-1]
            self.n_trajs = len(trajectories)

            self.horizon = horizon
            self.discount = discount ** np.arange(max_path_length, dtype=np.float32)
            self.discount = self.discount.reshape((-1, 1)).repeat(self.pref_dim, axis=-1)

            if avg:
                self.avg = np.arange(max_path_length, 0, -1, dtype=np.float32)
                self.avg = self.avg.reshape((-1, 1)).repeat(self.pref_dim, axis=-1)

            self.seq_obs = np.zeros((self.n_trajs, max_path_length, self.o_dim), dtype=np.float32)
            self.seq_act = np.zeros((self.n_trajs, max_path_length, self.a_dim), dtype=np.float32)
            self.seq_rew = np.zeros((self.n_trajs, max_path_length, self.pref_dim), dtype=np.float32)
            self.seq_pref = np.zeros((self.n_trajs, self.pref_dim), dtype=np.float32)
            self.seq_score = np.zeros((self.n_trajs, max_path_length, self.pref_dim), dtype=np.float32)
            seq_len = np.zeros((self.n_trajs,), dtype=np.int32)

            for i, traj in enumerate(trajectories):
                length = traj["observations"].shape[0]
                self.seq_obs[i, :length] = traj["observations"]
                self.seq_act[i, :length] = traj["actions"]
                self.seq_pref[i] = traj["preference"][0]
                self.seq_rew[i, :length] = traj["raw_rewards"]
                self.seq_score[i, :length] = traj["raw_rewards"]
                seq_len[i] = length

                if traj["terminals"][length-1]:
                    self.seq_score[i, length-1] -= terminal_penalty

                if gamma == 1:
                    self.seq_score[i, :length] = np.cumsum(traj["raw_rewards"][::-1], axis=0)[::-1]
                elif gamma > 0:
                    for j in range(length-2, -1, -1):
                        self.seq_score[i, j] += gamma * self.seq_score[i, j+1]
                
                if avg:
                    self.seq_score[i] /= self.avg

                # max_start = length - horizon
                # self.indices += [(i, start, start + horizon) for start in range(0, max_start + 1, 1)]
            flatten_obs = self.seq_obs.reshape(-1, self.o_dim)
            self.normalizers = {
                "state": GaussianNormalizer(flatten_obs)
            }
            # [:] result in a view rather than copy
            flatten_obs[:] = self.normalizers["state"].normalize(flatten_obs)

            sorted_index = np.argsort(self.seq_pref[:,0])
            self.seq_obs = self.seq_obs[sorted_index]
            self.seq_act = self.seq_act[sorted_index]
            self.seq_rew = self.seq_rew[sorted_index]
            self.seq_pref = self.seq_pref[sorted_index]
            self.seq_score = self.seq_score[sorted_index]
            seq_len = seq_len[sorted_index]

            if side:
                dropout_radius = int(self.n_trajs * dropout_percent / 2)
                dropout_centres = np.linspace(0, self.n_trajs-1, 2, dtype=np.int32)[1:-1]
                self.dropout_indices = [i for i in range(0, dropout_radius+1)] + [self.n_trajs-1-i for i in range(0, dropout_radius+1)]
            else:
                dropout_radius = int(self.n_trajs * dropout_percent / dropout_points / 2)
                dropout_centres = np.linspace(0, self.n_trajs-1, dropout_points+2, dtype=np.int32)[1:-1]
                self.dropout_indices = [i+j for i in dropout_centres for j in range(-dropout_radius, dropout_radius+1)]

            self.dropout_prefs = self.seq_pref[self.dropout_indices]
            self.seq_obs = np.delete(self.seq_obs, self.dropout_indices, axis=0)
            self.seq_act = np.delete(self.seq_act, self.dropout_indices, axis=0)
            self.seq_rew = np.delete(self.seq_rew, self.dropout_indices, axis=0)
            self.seq_pref = np.delete(self.seq_pref, self.dropout_indices, axis=0)
            self.seq_score = np.delete(self.seq_score, self.dropout_indices, axis=0)
            seq_len = np.delete(seq_len, self.dropout_indices, axis=0)

            self.n_trajs -= len(self.dropout_indices)

            self.indices = []
            for i in range(self.n_trajs):
                length = seq_len[i]
                max_start = length - horizon
                self.indices += [(i, start, start + horizon) for start in range(0, max_start + 1, 1)]
            
            self.seq_front = np.zeros((self.n_trajs,), dtype=np.bool8)
            sum_seq_rew = self.seq_rew.sum(1)

            self.front_idx = undominated_indices(sum_seq_rew)
            self.seq_front[self.front_idx] = True
            
            self.front_rew = sum_seq_rew[self.front_idx]
            self.front_pref = self.seq_pref[self.front_idx]

            if normalize_rewards:
                # st = TrajSparseTable(self.seq_score)

                if eps > 0:
                    # neighbor normalizing
                    r = 0
                    normalized_seq_score = np.zeros_like(self.seq_score)

                    for i in range(self.n_trajs):
                        tmp = self.seq_pref - self.seq_pref[i]
                        tmp = np.linalg.norm(tmp, 2, axis=1)
                        que = np.where(tmp < eps * np.sqrt(2))[0]

                        # brute force
                        neighbor_max_score = np.max(self.seq_score[que], axis=0)
                        neighbor_min_score = np.min(self.seq_score[que], axis=0)

                        normalized_seq_score[i] = (self.seq_score[i] - neighbor_min_score) / (neighbor_max_score - neighbor_min_score + 1e-6)
                    self.seq_score = normalized_seq_score

                else:
                    max_score = np.max(self.seq_score, axis=0)
                    min_score = np.min(self.seq_score, axis=0)
                    self.seq_score = (self.seq_score - min_score) / (max_score - min_score + 1e-6)
            
            if weighted_score:
                self.seq_score = self.seq_score * self.seq_pref[:, None, :]
        
        if not read_only:
            with open("dev/data/cache/LRUDatasetCache.pkl", "rb") as f:
                cache = pickle.load(file=f)
            
            cache.put((dataset, horizon, terminal_penalty, max_path_length, avg, gamma, normalize_rewards, eps, discount, weighted_score, dropout_percent, dropout_points, side), self.__dict__)

            with open("dev/data/cache/LRUDatasetCache.pkl", "wb") as f:
                pickle.dump(cache, file=f)
        print("Done.")

    def get_normalizer(self):
        return self.normalizers["state"]
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int):
        path_idx, start, end = self.indices[idx]

        scores = self.seq_score[path_idx, start:end]
        values = (scores * self.discount[:scores.shape[0]]).sum(0)

        data = {
            'obs': {
            'state': self.seq_obs[path_idx, start:end]},
            'act': self.seq_act[path_idx, start:end],
            'val': values,
            'pref': self.seq_pref[path_idx],
            'score': self.seq_score[path_idx, start:end]}
        
        # TODO: normalize val by horizon

        return dict_apply(data, torch.tensor)