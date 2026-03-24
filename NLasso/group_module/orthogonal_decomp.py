"""
组处理模块：正交分解
支持双变量和差变换、多变量PCA分解、自定义先验分解
对应paper 3.4节
"""
import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import qr
from ..base import _DTYPE, _COPY_WHEN_POSSIBLE


class OrthogonalDecomposer:
    """
    高相关变量组正交分解器
    支持三种分解模式：
    1. 双变量：和差变换（S=(X1+X2)/√2, D=(X1-X2)/√2）
    2. 多变量：PCA主成分分解（共同趋势 + 细节分量）
    3. 自定义先验：基于领域知识的指定载荷矩阵
    """
    def __init__(
        self,
        n_components: int = None,
        min_explained_variance_ratio: float = 0.7,
        custom_loadings: np.ndarray = None,
        random_state: int = 2026,
    ):
        self.n_components = n_components
        self.min_explained_variance_ratio = min_explained_variance_ratio
        self.custom_loadings = custom_loadings  # 自定义先验载荷矩阵 (k, m)
        self.random_state = random_state
        self.is_fitted_ = False

        # 拟合后属性
        self.group_size_ = None
        self.W_c_ = None  # 正交归一化载荷矩阵 (k, m)
        self.m_ = None  # 共同趋势分量数
        self.mean_ = None  # 组特征均值 (k,)
        self.std_ = None  # 组特征标准差 (k,)

    def fit(self, X_group: np.ndarray):
        """
        拟合正交分解
        Args:
            X_group: 组内特征矩阵 (n_samples, k_features)
        """
        n, k = X_group.shape
        self.group_size_ = k

        if k == 1:
            # 单变量无需分解
            self.is_fitted_ = True
            return self

        # 取消组内标准化：基类已经做了全局标准化，无需重复标准化
        self.mean_ = np.zeros(k, dtype=_DTYPE)
        self.std_ = np.ones(k, dtype=_DTYPE)
        X_std = X_group.astype(_DTYPE)

        if self.custom_loadings is not None:
            # 自定义先验载荷模式（paper 3.4.5节）
            W = self.custom_loadings.astype(_DTYPE)
            # QR正交化
            Q, _ = qr(W, mode='economic')
            self.W_c_ = Q
            self.m_ = Q.shape[1]

        elif k == 2:
            # 双变量和差变换（paper 3.4.1节）
            self.W_c_ = np.array([[1/ np.sqrt(2), 1/ np.sqrt(2)]], dtype=_DTYPE).T  # (2,1)
            self.m_ = 1

        else:
            # 多变量PCA分解（paper 3.4.2节）
            if self.n_components is None:
                # 自动选择主成分数，满足累计解释方差≥min_explained_variance_ratio
                pca = PCA(random_state=self.random_state)
                pca.fit(X_std)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                self.m_ = np.argmax(cumulative_variance >= self.min_explained_variance_ratio) + 1
                self.m_ = min(self.m_, k-1)  # 至少保留1个细节分量
            else:
                self.m_ = min(self.n_components, k-1)

            # 拟合PCA取前m个主成分
            pca = PCA(n_components=self.m_, random_state=self.random_state)
            pca.fit(X_std)
            self.W_c_ = pca.components_.T.astype(_DTYPE)  # (k, m)

        self.is_fitted_ = True
        return self

    def transform(self, X_group: np.ndarray) -> np.ndarray:
        """
        变换组特征为正交分量（共同趋势 + 细节分量）
        Args:
            X_group: 组内特征矩阵 (n_samples, k_features)
        Returns:
            X_transformed: 变换后的特征矩阵 (n_samples, m + k)
                前m列：共同趋势分量 T_c
                后k列：对冲细节分量 D_j
        """
        if not self.is_fitted_:
            raise RuntimeError("Decomposer not fitted yet.")

        n, k = X_group.shape
        if k != self.group_size_:
            raise ValueError(f"Expected group size {self.group_size_}, got {k}")

        if k == 1:
            # 单变量直接返回
            return X_group.astype(_DTYPE, copy=_COPY_WHEN_POSSIBLE)

        # 标准化
        X_std = (X_group - self.mean_) / self.std_

        # 计算共同趋势分量 T_c = X_std @ W_c (n, m)
        T_c = np.dot(X_std, self.W_c_)

        # 计算对冲细节分量 D_j = X_j - T_c @ (W_c^T e_j) (n, k)
        # 等价于 X_std @ (I - W_c W_c^T)
        D = X_std - np.dot(T_c, self.W_c_.T)

        # 拼接共同趋势和细节分量 (n, m + k)
        X_transformed = np.column_stack([T_c, D]).astype(_DTYPE)
        return X_transformed

    def fit_transform(self, X_group: np.ndarray) -> np.ndarray:
        return self.fit(X_group).transform(X_group)

    def inverse_transform(self, theta: np.ndarray) -> np.ndarray:
        """
        将变换空间的系数还原为原始组特征系数（paper 3.4.2节公式）
        Args:
            theta: 变换空间系数 (m + k,)
                前m个：共同趋势分量系数 gamma_c
                后k个：细节分量系数 gamma_d
        Returns:
            beta_group: 原始组特征系数 (k,)
        """
        if not self.is_fitted_:
            raise RuntimeError("Decomposer not fitted yet.")

        if self.group_size_ == 1:
            return theta.astype(_DTYPE, copy=_COPY_WHEN_POSSIBLE)

        # 拆分共同趋势系数和细节系数
        gamma_c = theta[:self.m_]
        gamma_d = theta[self.m_:]

        # 还原公式：beta = W_c @ gamma_c + gamma_d，再逆标准化
        beta_std = np.dot(self.W_c_, gamma_c) + gamma_d
        beta_group = beta_std / self.std_

        return beta_group.astype(_DTYPE)
